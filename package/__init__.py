"""
CyberFusion-IDS - Intrusion Detection System
Machine Learning models for intrusion detection
"""

__version__ = '1.0.0'
__author__ = 'darine'
__path__ = __path__  # This helps with package discovery

# Package info
PACKAGE_DIR = __file__.replace('/__init__.py', '') if '__file__' in locals() else '/home/darine/Desktop/CyberFusion-IDS'

def get_scripts():
    """Get all available script names"""
    import os
    scripts = [f for f in os.listdir(PACKAGE_DIR) 
               if f.endswith('.py') and f != '__init__.py']
    return sorted(scripts)

def run_script(script_name):
    """Run a specific script by name"""
    import os
    import sys
    
    if not script_name.endswith('.py'):
        script_name += '.py'
    
    script_path = os.path.join(PACKAGE_DIR, script_name)
    
    if os.path.exists(script_path):
        print(f"Running {script_name}...")
        with open(script_path, 'r') as f:
            exec(f.read(), {'__name__': '__main__'})
    else:
        print(f"Script {script_name} not found!")
        print(f"Available scripts: {get_scripts()}")

if __name__ == "__main__":
    print(f"CyberFusion-IDS v{__version__}")
    print(f"Available scripts: {get_scripts()}")
