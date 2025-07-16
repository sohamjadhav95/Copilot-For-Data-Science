# 🔧 Copilot for Data Science — Setup Guide

Welcome to the **Copilot for Data Science** project! This guide will walk you through the exact steps to set up and run the full application on your local machine, including launching the CLI Copilot with one click.

---

## 📦 Prerequisites

Ensure you have the following installed:

| Tool   | Version         | Notes                                                        |
| ------ | --------------- | ------------------------------------------------------------ |
| Python | 3.10+           | Install from [python.org](https://www.python.org/downloads/) |
| pip    | Latest          | Comes with Python                                            |
| Git    | Latest          | [git-scm.com](https://git-scm.com/)                          |
| ttyd   | Precompiled EXE | Terminal UI in browser                                       |
| ngrok  | Free account    | For public browser access                                    |

---

## 📁 1. Download the Project

You can either **clone the repository** or **download the latest version from Google Drive**:

### Option 1: Clone via GitHub (Use Latest Version)

```bash
git clone https://github.com/yourusername/Copilot-For-Data-Science.git
cd Copilot-For-Data-Science
```

### Option 2: Download ZIP (Latest Version)

* [📥 Download Latest Release via Google Drive](https://drive.google.com/drive/folders/1iv-jRfSXg-UgUjch_kZN95Nm4ARTqDon?usp=drive_link)
* Unzip the folder to a directory of your choice

---

## 📥 2. Install Python Dependencies

```bash
pip install -r requirements.txt
```

Make sure to clean up or confirm packages in `requirements.txt`. It typically includes:

```txt
rich
pandas
scikit-learn
matplotlib
groq
```

---

## 🔐 3. Set Up API Keys

The project uses the **Groq API** for code generation and explanations.

* Replace the placeholder Groq API key in your code files with your actual API key.
* (Optional) Create a `.env` file and load it using `python-dotenv`.

Example `.env`:

```env
GROQ_API_KEY=your-api-key-here
```

---

## 🧾 4. Prepare Your Dataset

* Place your dataset inside the data folder (or wherever expected).
* Update the dataset path inside `Core_Automation_Engine/Data.py` accordingly:

```python
filepath = lambda: "./data/your_dataset.csv"
```

---

## 🚀 5. One-Click Launch via .bat File

We’ve included a `.bat` file for Windows users:

### ✅ `launch_copilot.bat` — Runs your terminal UI

```bat
@echo off
cd /d %~dp0
start "" "C:\Path\To\ttyd.exe" python rich_cli.py
timeout /t 2 >nul
start "" "C:\Path\To\ngrok.exe" http 7681
```

> Edit this file to match your `ttyd.exe` and `ngrok.exe` paths.

### ✅ `Launch Copilot.lnk`

* This is a shortcut with a custom icon.
* You can double-click it just like a native app.

---

## 🌐 6. Run and Interact

After double-clicking the shortcut or running the `.bat` file manually, you’ll:

* Open a terminal session in your browser
* Be able to type natural language commands
* Receive output from the AI Copilot

---

## 💬 Example Commands to Try

| Operation        | Example Command                                |
| ---------------- | ---------------------------------------------- |
| Display Data     | "Show me the first 10 rows of the dataset"     |
| Modify Data      | "Remove all rows with null values"             |
| Visualize Data   | "Plot the distribution of the target variable" |
| Analyze Data     | "Generate a summary of the dataset"            |
| Machine Learning | "Build and test a machine learning model"      |

---

## 📁 Project Structure

```
Copilot-For-Data-Science/
├── launch_copilot.bat
├── Launch Copilot.lnk
├── copilot_icon.ico
├── requirements.txt
├── rich_cli.py
├── README.md
├── setup.md
│
├── Core_Automation_Engine/
├── Machine_Learning/
├── Retrival_Agumented_Generation/
└── data/
```

---

## 💡 Troubleshooting

* ❌ `Python not found` → Ensure Python is installed and added to PATH
* ❌ `Module not found` → Rerun `pip install -r requirements.txt`
* ❌ `Groq API error` → Check if your API key is valid and active

---

## 🙋 Need Help?

Reach out to us:

* 📧 [soham.ai.engineer@gmail.com](mailto:soham.ai.engineer@gmail.com)
* 📧 [omkargadakh1272004@gmail.com](mailto:omkargadakh1272004@gmail.com)

---

## ✅ You're Ready!

You now have a terminal-based Copilot AI running locally, with rich CLI and browser access. Happy building! 🚀
