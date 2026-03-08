# This is the WSGI configuration file for PythonAnywhere
# Save this as: /var/www/<yourusername>_pythonanywhere_com_wsgi.py

import sys
import os

# Add your project directory to the sys.path
path = '/home/<yourusername>/customerchurnprediction'
if path not in sys.path:
    sys.path.insert(0, path)

# Change to the project directory
os.chdir(path)

# Import your Flask app
from app import app as application
