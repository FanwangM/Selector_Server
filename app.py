# import json
import inspect
import os
import shutil
import tempfile
import threading
import uuid
import warnings
from datetime import datetime
from typing import Callable, Dict

import markdown
import matplotlib.pyplot as plt
import numpy as np
import orjson
import pandas as pd
from flask import Flask, Response, render_template, request, send_file
from flask_status import FlaskStatus
from selector.methods.distance import DISE, MaxMin, MaxSum, OptiSim
from selector.methods.partition import GridPartition, Medoid
from selector.methods.similarity import NSimilarity
from sklearn.metrics import pairwise_distances
from werkzeug.utils import secure_filename

try:
    from celery_config import celery

    CELERY_AVAILABLE = True
except ImportError:
    CELERY_AVAILABLE = False

app = Flask(__name__)
app_status = FlaskStatus(app)
app.config["MAX_CONTENT_LENGTH"] = 32 * 1024 * 1024  # 32MB max file size
app.config["UPLOAD_FOLDER"] = "uploads"
file_lock = threading.Lock()

# Ensure upload directory exists
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

ALLOWED_EXTENSIONS = {"txt", "npz", "xlsx", "xls"}

# Map algorithm names to their functions
SELECTION_ALGORITHM_MAP = {
    # Distance-based methods
    "MaxMin": MaxMin,
    "MaxSum": MaxSum,
    "OptiSim": OptiSim,
    "DISE": DISE,
    # Partition-based methods
    "GridPartition": GridPartition,
    # Similarity-based methods
    "NSimilarity": NSimilarity,
}


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def get_unique_upload_dir():
    """Create a unique directory for each upload session."""
    unique_dir = os.path.join(app.config["UPLOAD_FOLDER"], str(uuid.uuid4()))
    os.makedirs(unique_dir, exist_ok=True)
    return unique_dir


def clean_upload_dir(directory):
    """Safely clean up upload directory."""
    try:
        if os.path.exists(directory):
            shutil.rmtree(directory)
    except Exception as e:
        print(f"Error cleaning directory {directory}: {e}")


def load_data(filepath):
    """Load data from various file formats."""
    try:
        ext = filepath.rsplit(".", 1)[1].lower()
        if ext == "npz":
            with np.load(filepath) as data:
                return data["arr_0"] if "arr_0" in data else next(iter(data.values()))
        elif ext == "txt":
            return np.loadtxt(filepath)
        elif ext in ["xlsx", "xls"]:
            df = pd.read_excel(filepath)
            return df.to_numpy()
    except Exception as e:
        raise ValueError(f"Error loading file {filepath}: {str(e)}")


def save_data(data, format_type):
    """Save data in the specified format."""
    temp_dir = tempfile.mkdtemp()
    filename = os.path.join(temp_dir, f"result.{format_type}")

    if format_type == "npz":
        np.savez(filename, result=data)
    elif format_type == "txt":
        np.savetxt(filename, data)
    elif format_type in ["xlsx", "xls"]:
        pd.DataFrame(data).to_excel(filename, index=False)

    return filename


def create_json_response(data, status=200):
    """Create a JSON response using orjson for better numpy array handling"""
    return Response(
        orjson.dumps(data, option=orjson.OPT_SERIALIZE_NUMPY, default=str),
        status=status,
        mimetype="application/json",
    )


def read_markdown_file(filename):
    """Read and convert markdown file to HTML."""
    filepath = os.path.join(os.path.dirname(__file__), "md_files", filename)
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()

            # Pre-process math blocks to protect them
            # content = content.replace('\\\\', '\\\\\\\\')  # Escape backslashes in math

            # Convert markdown to HTML with math and table support
            md = markdown.Markdown(extensions=["tables", "fenced_code", "codehilite", "attr_list"])

            # First pass: convert markdown to HTML
            html = md.convert(content)

            # Post-process math blocks
            # Handle display math ($$...$$)
            html = html.replace("<p>$$", '<div class="math-block">$$')
            html = html.replace("$$</p>", "$$</div>")

            # Handle inline math ($...$)
            # We don't need special handling for inline math as MathJax will handle it

            return html
    except Exception as e:
        print(f"Error reading markdown file {filename}: {e}")
        return f"<p>Error loading content: {str(e)}</p>"


def get_default_parameters(func):
    """Get default parameters for a function from its signature."""
    sig = inspect.signature(func)
    defaults = {}

    for name, param in sig.parameters.items():
        if name == "self" or name == "fun_dist":  # Skip self and dist_metric
            continue
        if param.default is not param.empty:
            defaults[name] = param.default

    return defaults


@app.route("/get_default_params/<algorithm>")
def get_default_params(algorithm):
    """API endpoint to get default parameters for an algorithm."""
    if algorithm not in SELECTION_ALGORITHM_MAP:
        return create_json_response({"error": f"Unknown algorithm: {algorithm}"}, 400)

    try:
        # Get the algorithm class
        algorithm_class = SELECTION_ALGORITHM_MAP[algorithm]
        # Get default parameters from __init__
        params = get_default_parameters(algorithm_class.__init__)
        return create_json_response(params)
    except Exception as e:
        return create_json_response({"error": f"Error getting parameters: {str(e)}"}, 500)


@app.route("/get_default_selection_params/<algorithm>")
def get_default_selection_params(algorithm):
    """API endpoint to get default parameters for a selection algorithm."""
    if algorithm not in SELECTION_ALGORITHM_MAP:
        return create_json_response({"error": f"Algorithm unsupported: {algorithm}"}, 400)

    try:
        return create_json_response(get_default_selection_params(algorithm))
    except Exception as e:
        return create_json_response({"error": f"Error getting parameters: {str(e)}"}, 500)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/md/<filename>")
def get_markdown(filename):
    """Serve markdown files as HTML."""
    if not filename.endswith(".md"):
        filename = filename + ".md"
    html = read_markdown_file(filename)
    return create_json_response({"html": html})


def process_selection(arr, algorithm, parameters, dist_metric):
    """
    Process feature matrix using the specified selection algorithm.

    Parameters
    ----------
    arr : np.ndarray
        Input feature matrix
    algorithm : str
        Name of the selection algorithm to use
    parameters : dict
        Parameters for the algorithm
    dist_metric : str, optional
        Distance function to use.

    Returns
    -------
    dict
        Dictionary containing results and any warnings
    """
    warnings.filterwarnings('error')  # Convert warnings to exceptions
    result = {"success": False, "error": None, "warnings": [], "indices": None}

    try:
        print(f"Debug - Input parameters: algorithm={algorithm}, parameters={parameters}, dist_metric={dist_metric}")
        print(f"Debug - Array shape: {arr.shape}")

        # Get the algorithm class
        algorithm_class = SELECTION_ALGORITHM_MAP.get(algorithm)
        if algorithm_class is None:
            raise ValueError(f"Unknown algorithm: {algorithm}")

        # Get size parameter
        size = parameters.pop('size', None)
        if size is None:
            raise ValueError("Subset size must be specified")

        try:
            size = int(size)
            if size < 1:
                raise ValueError
        except (TypeError, ValueError):
            raise ValueError("Subset size must be a positive integer")

        print(f"Debug - Size parameter: {size}")

        # Validate size against array dimensions
        if size > arr.shape[0]:
            raise ValueError(f"Subset size ({size}) cannot be larger than the number of samples ({arr.shape[0]})")

        # Initialize algorithm
        print(f"Debug - Initializing algorithm with parameters: {parameters}")

        # Handle distance-based methods differently
        is_distance_based = algorithm in ["MaxMin", "MaxSum", "OptiSim", "DISE"]

        if is_distance_based:
            # For distance-based methods, always compute the distance matrix
            metric = dist_metric if dist_metric and dist_metric != "" else "euclidean"
            print(f"Debug - Computing distance matrix with metric: {metric}")
            arr_dist = pairwise_distances(arr, metric=metric)
            print(f"Debug - Distance matrix shape: {arr_dist.shape}")
        else:
            # For non-distance-based methods, use the original array
            print("Debug - Using original array for non-distance-based method")
            arr_dist = arr

        # Initialize and run the algorithm
        collector = algorithm_class(**parameters)
        indices = collector.select(arr_dist, size=size)

        print(f"Debug - Selected indices: {indices}")
        result["success"] = True
        result["indices"] = indices.tolist() if isinstance(indices, np.ndarray) else indices

    except Warning as w:
        print(f"Debug - Warning caught: {str(w)}")
        result["warnings"].append(str(w))
    except Exception as e:
        print(f"Debug - Error in process_selection: {str(e)}")
        print(f"Debug - Error type: {type(e)}")
        import traceback
        print(f"Debug - Traceback: {traceback.format_exc()}")
        result["error"] = str(e)

    return result


@app.route("/upload_selection", methods=["POST"])
def upload_selection_file():
    """Handle file upload and process selection."""
    try:
        print("Debug - Starting upload_selection_file")

        if "file" not in request.files:
            return create_json_response({"error": "No file provided"}, 400)

        file = request.files["file"]
        if file.filename == "":
            return create_json_response({"error": "No file selected"}, 400)

        if not allowed_file(file.filename):
            return create_json_response({"error": "File type not allowed"}, 400)

        # Get parameters
        algorithm = request.form.get("algorithm")
        if not algorithm:
            return create_json_response({"error": "No algorithm specified"}, 400)

        print(f"Debug - Algorithm: {algorithm}")

        # Get size parameter
        size = request.form.get("size")
        if not size:
            return create_json_response({"error": "Subset size must be specified"}, 400)

        print(f"Debug - Size: {size}")

        # Get distance function
        dist_metric = request.form.get("func_dist", "")
        print(f"Debug - Distance function: {dist_metric}")

        # Parse parameters
        try:
            parameters = orjson.loads(request.form.get("parameters", "{}"))
            print(f"Debug - Parsed parameters: {parameters}")
        except Exception as e:
            print(f"Debug - Error parsing parameters: {e}")
            parameters = {}

        # Add size to parameters
        parameters['size'] = size

        # Create a unique directory for this upload
        upload_dir = get_unique_upload_dir()
        print(f"Debug - Created upload directory: {upload_dir}")

        try:
            # Save file with unique name
            file_path = os.path.join(
                upload_dir, secure_filename(str(uuid.uuid4()) + "_" + file.filename)
            )
            print(f"Debug - Saving file to: {file_path}")

            with file_lock:
                file.save(file_path)

            # Load data
            array = load_data(file_path)
            print(f"Debug - Loaded array with shape: {array.shape}")

            # Process the selection with separate dist_metric parameter
            result = process_selection(array, algorithm, parameters, dist_metric)
            print(f"Debug - Process selection result: {result}")

            return create_json_response(result)

        except Exception as e:
            print(f"Debug - Error in file processing: {str(e)}")
            return create_json_response({"error": str(e)}, 500)

        finally:
            # Clean up the unique upload directory
            clean_upload_dir(upload_dir)

    except Exception as e:
        print(f"Debug - Error in upload_selection_file: {str(e)}")
        return create_json_response({"error": f"Error processing request: {str(e)}"}, 400)


@app.route("/download", methods=["POST"])
def download():
    try:
        data = orjson.loads(request.form["data"])
        format_type = request.form["format"]

        # Create temporary file
        temp_dir = tempfile.mkdtemp()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"selected_indices_{timestamp}"

        if format_type == "npz":
            filepath = os.path.join(temp_dir, f"{filename}.npz")
            np.savez(filepath, indices=np.array(data))
        elif format_type == "xlsx":
            filepath = os.path.join(temp_dir, f"{filename}.xlsx")
            pd.DataFrame(data, columns=["Index"]).to_excel(filepath, index=False)
        else:  # txt
            filepath = os.path.join(temp_dir, f"{filename}.txt")
            np.savetxt(filepath, np.array(data), fmt="%d")

        return send_file(filepath, as_attachment=True)
    except Exception as e:
        return create_json_response({"error": str(e)}, 500)
    finally:
        # Clean up temporary directory after sending file
        try:
            shutil.rmtree(temp_dir)
        except Exception:
            pass


@app.route("/status")
def server_status():
    """Return server status"""
    status = {
        "status": "ok",
        "message": "Server is running",
        "timestamp": datetime.now().isoformat(),
        "components": {"flask": True, "celery": False, "redis": False},
    }

    if CELERY_AVAILABLE:
        # Check Celery
        try:
            celery.control.ping(timeout=1)
            status["components"]["celery"] = True
        except Exception as e:
            print(f"Celery check failed: {e}")

        # Check Redis
        try:
            redis_client = celery.backend.client
            redis_client.ping()
            status["components"]["redis"] = True
        except Exception as e:
            print(f"Redis check failed: {e}")

        # Set overall status
        if not all(status["components"].values()):
            status["status"] = "degraded"
            status["message"] = "Some components are not available"
    else:
        status["message"] = "Running without Celery/Redis support"

    return create_json_response(status)


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8008)
