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
import numpy as np
import orjson
import pandas as pd
from selector.methods.distance import MaxMin, MaxSum, OptiSim, DISE
from selector.methods.partition import GridPartition, Medoid
from selector.methods.similarity import NSimilarity

import matplotlib.pyplot as plt
import numpy as np
import orjson
import pandas as pd
from flask import Flask, Response, render_template, request, send_file
from flask_status import FlaskStatus
from procrustes import (
    generalized,
    generic,
    kopt_heuristic_double,
    kopt_heuristic_single,
    orthogonal,
    orthogonal_2sided,
    permutation,
    permutation_2sided,
    rotational,
    softassign,
    symmetric,
)
from werkzeug.utils import secure_filename

from celery_config import celery

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
    """
    Collect the default arguments of a given function as a dictionary.

    Parameters
    ----------
    func : Callable
        The function to inspect.

    Returns
    -------
    Dict[str, object]
        A dictionary where keys are parameter names and values are their default values.

    """
    signature = inspect.signature(func)
    return {
        name: param.default
        for name, param in signature.parameters.items()
        if param.default is not inspect.Parameter.empty
    }


@app.route("/get_default_params/<algorithm>")
def get_default_params(algorithm):
    """API endpoint to get default parameters for an algorithm."""
    if algorithm not in ALGORITHM_MAP:
        return create_json_response({"error": f"Unknown algorithm: {algorithm}"}, 400)

    try:
        func = ALGORITHM_MAP[algorithm]
        return create_json_response(get_default_parameters(func))
    except Exception as e:
        return create_json_response({"error": f"Error getting parameters: {str(e)}"}, 500)


@app.route("/get_default_selection_params/<algorithm>")
def get_default_selection_params(algorithm):
    """API endpoint to get default parameters for a selection algorithm."""
    if algorithm not in SELECTION_ALGORITHM_MAP:
        return create_json_response({"error": f"Algorithm unsupported: {algorithm}"}, 400)

    try:
        func = SELECTION_ALGORITHM_MAP[algorithm]
        return create_json_response(get_default_parameters(func))
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


def process_procrustes(array1, array2, algorithm, parameters):
    """
    Process two arrays using the specified Procrustes algorithm.

    Parameters
    ----------
    array1 : np.ndarray
        First input array
    array2 : np.ndarray
        Second input array
    algorithm : str
        Name of the Procrustes algorithm to use
    parameters : dict
        Parameters for the algorithm

    Returns
    -------
    dict
        Dictionary containing results and any warnings
    """
    warning_message = None

    # Check for NaN values
    if np.isnan(array1).any() or np.isnan(array2).any():
        array1 = np.nan_to_num(array1)
        array2 = np.nan_to_num(array2)
        warning_message = "Input matrices contain NaN values. Replaced with 0."

    # Apply Procrustes algorithm
    if algorithm.lower() in ALGORITHM_MAP:
        result = ALGORITHM_MAP[algorithm.lower()](array1, array2, **parameters)
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")

    # Extract results
    transformation = (
        result.t
        if hasattr(result, "t")
        else result.t1 if hasattr(result, "t1") else np.eye(array1.shape[1])
    )

    new_array = (
        result.new_array
        if hasattr(result, "new_array")
        else result.array_transformed if hasattr(result, "array_transformed") else array2
    )

    # Prepare response
    response_data = {
        "error": float(result.error),
        "transformation": transformation,
        "new_array": new_array,
    }

    if warning_message:
        response_data["warning"] = warning_message

    return response_data


def process_selection(array, algorithm, parameters):
    """
    Process feature matrix using the specified selection algorithm.

    Parameters
    ----------
    array : np.ndarray
        Input feature matrix
    algorithm : str
        Name of the selection algorithm to use
    parameters : dict
        Parameters for the algorithm

    Returns
    -------
    dict
        Dictionary containing results and any warnings
    """
    warnings.filterwarnings('error')  # Convert warnings to exceptions
    result = {"success": False, "error": None, "warnings": [], "indices": None}

    try:
        # Get the algorithm class
        algorithm_class = SELECTION_ALGORITHM_MAP.get(algorithm)
        if algorithm_class is None:
            raise ValueError(f"Unknown algorithm: {algorithm}")

        # Initialize and run the algorithm
        selector = algorithm_class(**parameters)
        indices = selector.select(array)

        result["success"] = True
        result["indices"] = indices.tolist()

    except Warning as w:
        result["warnings"].append(str(w))
    except Exception as e:
        result["error"] = str(e)

    return result


@celery.task(bind=True)
def process_matrices(self, algorithm, params, matrix1_data, matrix2_data):
    """Celery task for processing matrices asynchronously."""
    try:
        # Convert lists back to numpy arrays
        matrix1 = np.asarray(matrix1_data, dtype=float)
        matrix2 = np.asarray(matrix2_data, dtype=float)

        if matrix1.size == 0 or matrix2.size == 0:
            raise ValueError("Empty matrix received")

        return process_procrustes(matrix1, matrix2, algorithm, params)

    except Exception as e:
        return {"error": f"Processing error: {str(e)}"}


@app.route("/upload", methods=["POST"])
def upload_file():
    """Handle file upload and process matrices."""
    print("Received upload request")

    if "file1" not in request.files or "file2" not in request.files:
        return create_json_response({"error": "Both files are required"}, 400)

    file1 = request.files["file1"]
    file2 = request.files["file2"]
    algorithm = request.form.get("algorithm", "orthogonal")

    if file1.filename == "" or file2.filename == "":
        return create_json_response({"error": "No selected files"}, 400)

    if not (allowed_file(file1.filename) and allowed_file(file2.filename)):
        return create_json_response({"error": "Invalid file type"}, 400)

    # Create a unique directory for this upload
    upload_dir = get_unique_upload_dir()

    try:
        # Parse parameters
        try:
            parameters = orjson.loads(request.form.get("parameters", "{}"))
        except orjson.JSONDecodeError:
            parameters = get_default_parameters(algorithm)

        # Save files with unique names
        file1_path = os.path.join(
            upload_dir, secure_filename(str(uuid.uuid4()) + "_" + file1.filename)
        )
        file2_path = os.path.join(
            upload_dir, secure_filename(str(uuid.uuid4()) + "_" + file2.filename)
        )

        with file_lock:
            file1.save(file1_path)
            file2.save(file2_path)

        # Load data
        array1 = load_data(file1_path)
        array2 = load_data(file2_path)
        print(f"Arrays loaded - shapes: {array1.shape}, {array2.shape}")

        # Process the matrices
        result = process_procrustes(array1, array2, algorithm, parameters)
        return create_json_response(result)

    except Exception as e:
        print(f"Error occurred: {str(e)}")
        import traceback

        print(traceback.format_exc())
        return create_json_response({"error": str(e)}, 500)

    finally:
        # Clean up the unique upload directory
        clean_upload_dir(upload_dir)


@app.route("/upload_selection", methods=["POST"])
def upload_selection_file():
    """Handle file upload and process selection."""
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

    try:
        parameters = orjson.loads(request.form.get("parameters", "{}"))
    except Exception as e:
        return create_json_response({"error": f"Invalid parameters: {str(e)}"}, 400)

    # Create unique upload directory
    upload_dir = get_unique_upload_dir()

    try:
        # Save and process file
        filename = secure_filename(file.filename)
        filepath = os.path.join(upload_dir, filename)
        file.save(filepath)

        # Load data
        array = load_data(filepath)

        # Process selection
        result = process_selection(array, algorithm, parameters)

        if result["success"]:
            # Save indices for later download
            indices_path = os.path.join(upload_dir, "indices.txt")
            np.savetxt(indices_path, result["indices"], fmt="%d")

            return create_json_response({
                "success": True,
                "warnings": result["warnings"],
                "indices": result["indices"],
                "upload_dir": upload_dir
            })
        else:
            return create_json_response({
                "success": False,
                "error": result["error"],
                "warnings": result["warnings"]
            })

    except Exception as e:
        return create_json_response({"error": str(e)}, 500)
    finally:
        # Clean up
        clean_upload_dir(upload_dir)


@app.route("/download/<upload_dir>")
def download(upload_dir):
    """Download selected indices."""
    try:
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], upload_dir, "indices.txt")
        return send_file(filepath, as_attachment=True, download_name="selected_indices.txt")
    except Exception as e:
        return create_json_response({"error": str(e)}, 500)


@app.route("/status/<task_id>")
def task_status(task_id):
    task = process_matrices.AsyncResult(task_id)
    if task.state == "PENDING":
        response = {"state": task.state, "status": "Pending..."}
    elif task.state != "FAILURE":
        response = {
            "state": task.state,
            "result": task.result,
        }
        if task.state == "SUCCESS":
            response["status"] = "Task completed!"
        else:
            response["status"] = "Processing..."
    else:
        response = {
            "state": task.state,
            "status": str(task.info),
        }
    return create_json_response(response)


@app.route("/status")
def server_status():
    """Return server status"""
    status = {"status": "ok", "components": {"flask": True, "celery": False, "redis": False}}

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

    # Set overall status based on components
    if not all(status["components"].values()):
        status["status"] = "degraded"

    return create_json_response(status)


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8008)
