import tkinter as tk
import sys
import traceback
import os
import threading
import io
from datetime import datetime
import textwrap
import customtkinter as ctk
from PIL import Image, ImageTk
import json
import time
from tkinterdnd2 import DND_FILES, TkinterDnD

# Set appearance mode and default color theme
ctk.set_appearance_mode("System")  # "System", "Dark", "Light"
ctk.set_default_color_theme("blue")  # "blue", "green", "dark-blue"

# Ensure all required paths are set
sys.path.append(r"E:\Projects\Copilot-For-Data-Science\Copilote for data science\Phase 3 Human in the Loop (Interactive AI)\v3.5 Fundamental Changes\Machine_Learning")
sys.path.append(r"E:\Projects\Copilot-For-Data-Science\Copilote for data science\Phase 3 Human in the Loop (Interactive AI)\v3.5 Fundamental Changes\Core_Automation_Engine")
sys.path.append(r"E:\Projects\Copilot-For-Data-Science\Copilote for data science\Phase 3 Human in the Loop (Interactive AI)\v3.5 Fundamental Changes\Retrival_Agumented_Generation")

# Import modules - Display friendly error if imports fail
try:
    from NL_processor import NL_processor, split_multi_commands, genral_response_chatbot
    from Engine_Display_data import Groq_Input as Display_Groq_Input
    from Engine_Visualize_data import Groq_Input as Visualize_Groq_Input
    from Engine_Modify_data import Groq_Input as Modify_Groq_Input, undo_last_change
    from Engine_Data_analysis import DataAnalysisReport, Dashboard
    from rag_command_parser import basic_rag
    from ML_Models_Engine_autogluon import SupervisedUniversalMachineLearning
    from groq import Groq
    from Data import update_data_path, import_data
except ImportError as e:
    print(f"Import Error: {e}. Please check that all required modules are installed and paths are correct.")
    ctk.CTk().withdraw()  # Prevent empty window
    ctk.CTkMessageBox.showerror("Import Error", 
                               f"Error importing required modules: {e}\n\nPlease check that all required modules are installed and paths are correct.")
    sys.exit(1)

# Initialize Groq client with API key
api_key = "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"
client = Groq(api_key=api_key)


# Initialize ML model and other objects
ML = SupervisedUniversalMachineLearning()
dashboard = Dashboard()
data_analysis_report = DataAnalysisReport()

# Output capture class
class OutputCapture:
    def __init__(self):
        self.buffer = io.StringIO()
        self.stdout = sys.stdout
        
    def __enter__(self):
        sys.stdout = self.buffer
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout = self.stdout
        
    def get_output(self):
        return self.buffer.getvalue()

def explain_error(error_message):
    """Use Groq API to dynamically explain errors"""
    try:
        prompt = f"""
        You are an AI assistant that helps developers debug Python errors.
        Given the following error message, provide an explanation and a solution.

        Error Message:
        {error_message}

        Explanation and Suggested Fix:
        """

        completion = client.chat.completions.create(
            model="gemma2-9b-it",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.4,
            max_tokens=1024,
            top_p=0.95,
            stream=False,
            stop=None,
        )

        return completion.choices[0].message.content.strip()
    except Exception:
        return "Error explanation service is unavailable. Please check your API connection."

def safe_execute(operation_function, *args):
    """Safely executes any function with error handling and captures output."""
    with OutputCapture() as output:
        try:
            result = operation_function(*args)
            return output.get_output(), None
        except Exception as e:
            error_message = traceback.format_exc()
            explanation = explain_error(error_message)
            error_output = f"\n[ERROR] An error occurred:\n{explanation}"
            return output.get_output(), error_output

def route_command(operation, user_input):
    """Routes command execution based on operation type."""
    output = ""
    error = None
    
    refined_user_input_by_rag = basic_rag(user_input)
    operation_output = f"\nExecuting command: {refined_user_input_by_rag}\n"
    
    if operation == "visualize":
        operation_output += "Visualizing data...\n"
        func_output, func_error = safe_execute(Visualize_Groq_Input, refined_user_input_by_rag)
    elif operation == "display":
        operation_output += "Displaying data...\n"
        func_output, func_error = safe_execute(Display_Groq_Input, refined_user_input_by_rag)
    elif operation == "modify":
        operation_output += "Modifying data...\n"
        func_output, func_error = safe_execute(Modify_Groq_Input, refined_user_input_by_rag)
    elif operation == "undo":
        operation_output += "Undoing last modification...\n"
        func_output, func_error = safe_execute(undo_last_change)
    elif operation == "meaningful_response":
        operation_output += "Generating a response...\n"
        func_output, func_error = safe_execute(genral_response_chatbot, refined_user_input_by_rag)
    elif operation == "analyze_data":
        operation_output += "Analyzing data...\n"
        func_output, func_error = safe_execute(data_analysis_report.generate_insights, refined_user_input_by_rag)
    elif operation == "generate_report":
        operation_output += "Generating PDF report...\n"
        func_output, func_error = safe_execute(data_analysis_report.generate_pdf_report, refined_user_input_by_rag)
    elif operation == "create_dashboard":
        operation_output += "Creating dashboard...\n"
        func_output, func_error = safe_execute(dashboard.create_dashboard, refined_user_input_by_rag)
    elif operation == "build_model":
        operation_output += "Building ML model...\n"
        func_output, func_error = safe_execute(ML.build_model_and_test, refined_user_input_by_rag)
    elif operation == "test_model":
        operation_output += "Testing ML model...\n"
        func_output, func_error = safe_execute(ML.build_model_and_test, refined_user_input_by_rag)
    elif operation == "deploy_model":
        operation_output += "Deploying ML model...\n"
        func_output, func_error = safe_execute(ML.deploy_model, refined_user_input_by_rag)
    elif operation == "predict_custom_input":
        operation_output += "Predicting custom input...\n"
        func_output, func_error = safe_execute(ML.predict_custom_input, refined_user_input_by_rag)
    elif operation == "os_operations":
        operation_output += "Performing OS operations...\n"

    else:
        operation_output += "Unable to determine the operation. Please try again.\n"
        func_output = ""
        func_error = None
    
    output = operation_output + (func_output if func_output else "")
    error = func_error
    
    return output, error

def execute_multiple_commands(user_input):
    """Execute multiple commands sequentially with error handling."""
    all_output = ""
    all_errors = []
    
    commands = split_multi_commands(user_input)  # Split the input into commands
    for command in commands:
        if command:
            refined_user_input_by_rag = basic_rag(command)
            operation = NL_processor(refined_user_input_by_rag)
            output, error = route_command(operation, refined_user_input_by_rag)
            all_output += output + "\n"
            if error:
                all_errors.append(error)
    
    return all_output, all_errors

class SmoothScrollableFrame(ctk.CTkScrollableFrame):
    """Enhanced scrollable frame with smooth scrolling"""
    def __init__(self, master, **kwargs):
        super().__init__(master, **kwargs)
        self.bind_all("<MouseWheel>", self._on_mousewheel)

    def _on_mousewheel(self, event):
        # Smoother scrolling
        self._parent_canvas.yview_scroll(int(-1*(event.delta/120)), "units")

class MessageBubble(ctk.CTkFrame):
    """Custom message bubble widget for chat messages"""
    def __init__(self, master, sender, message, timestamp, **kwargs):
        if sender == "You":
            kwargs["fg_color"] = "#e3f2fd"  # Light blue for user
            text_color = "#303f9f"
        elif sender == "System":
            kwargs["fg_color"] = "#faf9f6"  # Light amber for system messages
            text_color = "#ff6f00"
        else:
            kwargs["fg_color"] = "#f1f8e9"  # Light green for assistant
            text_color = "#2e7d32"
            
        super().__init__(master, corner_radius=10, **kwargs)
        
        # Header layout
        header_frame = ctk.CTkFrame(self, fg_color="transparent")
        header_frame.pack(fill="x", padx=10, pady=(10, 0))
        
        sender_label = ctk.CTkLabel(header_frame, text=sender, 
                                   font=ctk.CTkFont(family="Segoe UI", size=12, weight="bold"),
                                   text_color=text_color)
        sender_label.pack(side="left")
        
        time_label = ctk.CTkLabel(header_frame, text=timestamp,
                                 font=ctk.CTkFont(family="Segoe UI", size=10),
                                 text_color="#757575")
        time_label.pack(side="right")
        
        # Message content
        message_box = ctk.CTkTextbox(self, wrap="word", height=max(100, min(20 * len(message.split('\n')), 300)),
                                   font=ctk.CTkFont(family="Segoe UI", size=12),
                                   fg_color="transparent", activate_scrollbars=True,
                                   text_color="#212121")
        message_box.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Insert message text
        message_box.insert("1.0", message)
        message_box.configure(state="disabled")  # Make read-only

class PromptButton(ctk.CTkButton):
    """Custom styled button for command prompts"""
    def __init__(self, master, text, command, **kwargs):
        super().__init__(master, text=text, command=command, 
                        fg_color="#2979ff", hover_color="#1565c0",
                        corner_radius=8, **kwargs)

class DataScienceCopilotApp(TkinterDnD.Tk):
    def __init__(self):
        super().__init__()
        
        # App configuration
        self.title("Data Science Copilot")
        self.geometry("1200x800")
        self.minsize(900, 600)
        
        # Try to set app icon
        try:
            icon_path = "icon.ico"
            if os.path.exists(icon_path):
                self.iconbitmap(icon_path)
        except:
            pass
        
        # Configure the grid layout
        self.grid_columnconfigure(0, weight=0)  # Sidebar (fixed width)
        self.grid_columnconfigure(1, weight=1)  # Main area
        self.grid_rowconfigure(0, weight=1)
        
        # Create main components
        self.create_sidebar()
        self.create_main_panel()
        
        # Chat history
        self.chat_history = []
        
        # Set theme and appearance when starting
        self.theme_var = ctk.StringVar(value="System")
        self.appearance_mode_menu = None
        self.theme_button = None
        self.setup_theme_control()
        
        # Add typing animation effect variables
        self.typing_animation = False
        self.typing_index = 0
        self.typing_speed = 10  # ms between characters
        self.typing_message = ""
        self.typing_sender = ""
        self.current_typing_label = None
        
        # Add welcome message with typing animation
        self.animate_message("System", "Welcome to Data Science Copilot! Type your command below or use the templates from the sidebar.")
        
    def create_sidebar(self):
        """Create the sidebar with available commands and settings"""
        # Sidebar container with slight color difference
        self.sidebar = ctk.CTkFrame(self, fg_color=("gray90", "gray17"), width=240)
        self.sidebar.grid(row=0, column=0, sticky="nsew", padx=0, pady=0)
        self.sidebar.grid_propagate(False)  # Prevent sidebar from resizing
        
        # App logo and title at the top
        logo_frame = ctk.CTkFrame(self.sidebar, fg_color="transparent")
        logo_frame.pack(fill="x", padx=10, pady=10)
        
        # Logo (placeholder - replace with actual logo)
        try:
            # Try to load logo if it exists, otherwise use text
            # logo_img = Image.open("logo.png").resize((32, 32))
            # logo_photo = ImageTk.PhotoImage(logo_img)
            # logo_label = ctk.CTkLabel(logo_frame, image=logo_photo, text="")
            # logo_label.image = logo_photo  # Keep reference
            # logo_label.pack(side="left", padx=(0, 10))
            
            # App title
            title_label = ctk.CTkLabel(logo_frame, text="Data Science Copilot",
                                    font=ctk.CTkFont(family="Segoe UI", size=18, weight="bold"))
            title_label.pack(side="left", padx=10)
        except:
            # Fallback to text-only title
            title_label = ctk.CTkLabel(logo_frame, text="Data Science Copilot",
                                    font=ctk.CTkFont(family="Segoe UI", size=18, weight="bold"))
            title_label.pack(side="left", padx=10)
        
        # Create file drop zone
        self.create_file_drop_zone()
        
        # Create scrollable frame for command categories
        self.sidebar_scroll = SmoothScrollableFrame(self.sidebar, fg_color="transparent")
        self.sidebar_scroll.pack(fill="both", expand=True, padx=5, pady=5)
        
        # Command categories
        categories = {
            "Data Operations": [
                {"text": "Display Data", "command": "Show me the first 5 rows of the dataset", "icon": "📊"},
                {"text": "Visualize Data", "command": "Create a bar chart of ", "icon": "📈"},
                {"text": "Modify Data", "command": "Clean missing values in the ", "icon": "🔧"},
            ],
            "Analysis": [
                {"text": "Analyze Data", "command": "Analyze trends in the ", "icon": "🔍"},
                {"text": "Generate Report", "command": "Generate a PDF report of the analysis results", "icon": "📄"},
                {"text": "Create Dashboard", "command": "Create a dashboard showing ", "icon": "📱"},
            ],
            "Machine Learning": [
                {"text": "Build Model", "command": "Build a classification model to predict ", "icon": "🧠"},
                {"text": "Test Model", "command": "Test the model on ", "icon": "🧪"},
                {"text": "Deploy Model", "command": "Deploy the model as ", "icon": "🚀"},
                {"text": "Predict", "command": "Predict using the model with inputs ", "icon": "✨"},
            ],
        }
        
        # Create command buttons organized by categories
        for category, commands in categories.items():
            # Category frame with heading
            category_frame = ctk.CTkFrame(self.sidebar_scroll, fg_color="transparent")
            category_frame.pack(fill="x", pady=5, padx=2)
            
            # Category heading
            ctk.CTkLabel(category_frame, text=category, 
                      font=ctk.CTkFont(family="Segoe UI", size=14, weight="bold")).pack(
                          fill="x", padx=5, pady=5)
            
            # Command buttons
            for cmd in commands:
                cmd_text = f"{cmd['icon']} {cmd['text']}"
                cmd_btn = PromptButton(
                    category_frame, text=cmd_text, height=32,
                    command=lambda c=cmd["command"]: self.insert_command_template(c))
                cmd_btn.pack(fill="x", padx=5, pady=3)
        
        # Quick actions at the bottom
        quick_actions_frame = ctk.CTkFrame(self.sidebar, fg_color="transparent")
        quick_actions_frame.pack(fill="x", side="bottom", padx=10, pady=10)
        
        # Actions buttons
        undo_btn = ctk.CTkButton(quick_actions_frame, text="🔄 Undo Last Change", 
                              fg_color="#ff7043", hover_color="#e64a19",
                              command=lambda: self.process_command("undo"))
        undo_btn.pack(fill="x", pady=5)
        
        clear_btn = ctk.CTkButton(quick_actions_frame, text="🗑️ Clear History", 
                               fg_color="#78909c", hover_color="#546e7a",
                               command=self.clear_history)
        clear_btn.pack(fill="x", pady=5)
    
    def setup_theme_control(self):
        """Setup theme control dropdown in the sidebar"""
        theme_frame = ctk.CTkFrame(self.sidebar, fg_color="transparent")
        theme_frame.pack(fill="x", side="bottom", padx=10, pady=5)
        
        # Theme switcher
        ctk.CTkLabel(theme_frame, text="Theme:", 
                  font=ctk.CTkFont(family="Segoe UI", size=12)).pack(side="left", padx=5)
        
        self.appearance_mode_menu = ctk.CTkOptionMenu(
            theme_frame, values=["System", "Light", "Dark"],
            variable=self.theme_var, width=100,
            command=self.change_appearance_mode)
        self.appearance_mode_menu.pack(side="right", padx=5)
    
    def change_appearance_mode(self, mode):
        """Change the app appearance mode"""
        ctk.set_appearance_mode(mode)
    
    def create_main_panel(self):
        """Create the main panel with chat interface"""
        # Main panel container
        self.main_panel = ctk.CTkFrame(self, corner_radius=0)
        self.main_panel.grid(row=0, column=1, sticky="nsew")
        
        # Configure rows to accommodate chat area and input area
        self.main_panel.grid_rowconfigure(0, weight=1)  # Chat area
        self.main_panel.grid_rowconfigure(1, weight=0)  # Input area
        self.main_panel.grid_columnconfigure(0, weight=1)
        
        # Chat area with smooth scrolling
        self.chat_area = SmoothScrollableFrame(self.main_panel, fg_color="transparent")
        self.chat_area.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        
        # Input area at the bottom
        self.create_input_area()
    
    def create_input_area(self):
        """Create the input area at the bottom of the main panel"""
        input_frame = ctk.CTkFrame(self.main_panel, fg_color="transparent")
        input_frame.grid(row=1, column=0, sticky="ew", padx=10, pady=10)
        input_frame.grid_columnconfigure(0, weight=1)
        
        # Text input with placeholder
        self.input_textbox = ctk.CTkTextbox(input_frame, height=80, corner_radius=10,
                                         font=ctk.CTkFont(family="Segoe UI", size=13))
        self.input_textbox.grid(row=0, column=0, sticky="ew", padx=(0, 10))
        self.input_textbox.bind("<Control-Return>", self.on_enter_key)
        
        # Send button
        self.send_button = ctk.CTkButton(input_frame, text="Send", width=100, height=36,
                                      corner_radius=10, fg_color="#2979ff", hover_color="#1565c0",
                                      command=self.on_send)
        self.send_button.grid(row=0, column=1, sticky="e")
        
        # Placeholder text and hint
        self.input_textbox.insert("1.0", "Type your message here...")
        self.input_textbox.bind("<FocusIn>", self.on_input_focus_in)
        self.input_textbox.bind("<FocusOut>", self.on_input_focus_out)
        
        # Hint label below input
        hint_label = ctk.CTkLabel(input_frame, text="Press Ctrl+Enter to send",
                                font=ctk.CTkFont(family="Segoe UI", size=10),
                                fg_color="transparent", text_color="#757575")
        hint_label.grid(row=1, column=0, sticky="w", pady=(2, 0))
    
    def on_input_focus_in(self, event):
        """Clear placeholder text when input receives focus"""
        if self.input_textbox.get("1.0", "end-1c") == "Type your message here...":
            self.input_textbox.delete("1.0", "end")
    
    def on_input_focus_out(self, event):
        """Restore placeholder text if input is empty"""
        if not self.input_textbox.get("1.0", "end-1c").strip():
            self.input_textbox.delete("1.0", "end")
            self.input_textbox.insert("1.0", "Type your message here...")
    
    def insert_command_template(self, template):
        """Insert command template into the input field"""
        self.input_textbox.delete("1.0", "end")
        self.input_textbox.insert("1.0", template)
        self.input_textbox.focus_set()
    
    def on_enter_key(self, event):
        """Handle Ctrl+Enter key press"""
        self.on_send()
        return "break"  # Prevents the default behavior
    
    def on_send(self):
        """Process the command when send button is clicked"""
        user_input = self.input_textbox.get("1.0", "end-1c").strip()
        if not user_input or user_input == "Type your message here...":
            return
            
        # Clear the input field
        self.input_textbox.delete("1.0", "end")
        
        # Add user message to chat
        self.add_message("You", user_input)
        
        # Temporarily disable the send button while processing
        self.send_button.configure(state="disabled", text="Processing...")
        
        # Process the command in a separate thread to keep UI responsive
        threading.Thread(target=self.process_command, args=(user_input,), daemon=True).start()
    
    def process_command(self, user_input):
        """Process the user command"""
        try:
            if user_input.lower() == "undo":
                # Special case for undo command
                output, error = safe_execute(undo_last_change)
                response = "Undoing last change...\n" + (output if output else "")
            elif "then" in user_input.lower() or "and" in user_input.lower() or "after" in user_input.lower():
                # Multiple commands
                output, errors = execute_multiple_commands(user_input)
                response = output
                if errors:
                    response += "\n\n" + "\n".join([e for e in errors if e])
            else:
                # Single command
                operation = NL_processor(user_input)
                output, error = route_command(operation, user_input)
                response = output
                if error:
                    response += "\n\n" + error
            
            # Add response to chat with typing animation
            self.after(500, lambda: self.animate_message("Copilot", response))
            
        except Exception as e:
            error_msg = f"Error processing command: {str(e)}\n{traceback.format_exc()}"
            self.after(0, lambda: self.add_message("Copilot", error_msg))
        
        # Re-enable the send button
        self.after(0, lambda: self.send_button.configure(state="normal", text="Send"))
    
    def animate_message(self, sender, message):
        """Start typing animation for a new message"""
        if self.typing_animation:
            # If already typing, finish the current message immediately
            if self.current_typing_label:
                self.current_typing_label.destroy()
                self.add_message(self.typing_sender, self.typing_message)
        
        # Set up for new animation
        self.typing_animation = True
        self.typing_message = message
        self.typing_sender = sender
        self.typing_index = 0
        
        # Create typing indicator
        timestamp = datetime.now().strftime("%H:%M:%S")
        
        # Create frame for the typing message
        if sender == "You":
            bg_color = "#e3f2fd"  # Light blue for user
        elif sender == "System":
            bg_color = "#faf9f6"  # Light amber for system messages
        else:
            bg_color = "#f1f8e9"  # Light green for assistant
            
        typing_frame = ctk.CTkFrame(self.chat_area, corner_radius=10, fg_color=bg_color)
        typing_frame.pack(fill="x", padx=10, pady=5)
        
        # Header layout
        header_frame = ctk.CTkFrame(typing_frame, fg_color="transparent")
        header_frame.pack(fill="x", padx=10, pady=(10, 0))
        
        if sender == "You":
            text_color = "#303f9f"
        elif sender == "System":
            text_color = "#ff6f00"
        else:
            text_color = "#2e7d32"
            
        sender_label = ctk.CTkLabel(header_frame, text=sender, 
                                  font=ctk.CTkFont(family="Segoe UI", size=12, weight="bold"),
                                  text_color=text_color)
        sender_label.pack(side="left")
        
        time_label = ctk.CTkLabel(header_frame, text=timestamp,
                                font=ctk.CTkFont(family="Segoe UI", size=10),
                                text_color="#757575")
        time_label.pack(side="right")
        
        # Create typing indicator content
        typing_content = ctk.CTkLabel(typing_frame, text="...", 
                                   font=ctk.CTkFont(family="Segoe UI", size=12),
                                   text_color="#212121", anchor="w", 
                                   justify="left", padx=10, pady=10)
        typing_content.pack(fill="x", padx=10, pady=10)
        
        self.current_typing_label = typing_frame
        
        # Start animation
        self.animate_typing(typing_content)
    
    def animate_typing(self, label):
        """Animate typing effect for a message"""
        if not self.typing_animation:
            return
            
        # Increment displayed text
        displayed_chars = min(self.typing_index, len(self.typing_message))
        displayed_text = self.typing_message[:displayed_chars]
        
        # Handle special case for very short messages
        if len(displayed_text) <= 3:
            displayed_text = displayed_text + "..." 
        
        # Update label text
        label.configure(text=displayed_text)
        
        # Auto-scroll to bottom
        self.chat_area._parent_canvas.update_idletasks()
        self.chat_area._parent_canvas.yview_moveto(1.0)
        
        # Progress animation or finish
        if self.typing_index < len(self.typing_message):
            self.typing_index += max(1, len(self.typing_message) // 100)  # Speed adjusts based on message length
            self.after(self.typing_speed, lambda: self.animate_typing(label))
        else:
            # Animation complete, replace with final message
            self.typing_animation = False
            if self.current_typing_label:
                self.current_typing_label.destroy()
                self.add_message(self.typing_sender, self.typing_message)
    
    def add_message(self, sender, message):
        """Add a message to the chat"""
        # Create timestamp
        timestamp = datetime.now().strftime("%H:%M:%S")
        
        # Create message bubble
        message_bubble = MessageBubble(self.chat_area, sender, message, timestamp)
        message_bubble.pack(fill="x", padx=10, pady=5)
        
        # Store in history
        self.chat_history.append((sender, message, timestamp))
        
        # Auto-scroll to bottom
        self.chat_area._parent_canvas.update_idletasks()
        self.chat_area._parent_canvas.yview_moveto(1.0)
        
        # Save chat history to file (optional)
        # self.save_chat_history()
    
    def clear_history(self):
        """Clear the chat history"""
        # Create confirmation dialog
        confirm = ctk.CTkMessageBox.askyesno(
            "Clear History", 
            "Are you sure you want to clear the entire chat history?")
        if not confirm:
            return

        # Destroy all message widgets
        for widget in self.chat_area.winfo_children():
            widget.destroy()

        # Clear history list
        self.chat_history = []

        # Add a system message to indicate history was cleared
        self.add_message("System", "Chat history cleared.")

    def save_chat_history(self):
        """Save chat history to a JSON file"""
        history_path = "chat_history.json"
        with open(history_path, "w") as file:
            json.dump(self.chat_history, file, indent=4)
        print(f"Chat history saved to {history_path}")

    def load_chat_history(self):
        """Load chat history from a JSON file"""
        history_path = "chat_history.json"
        if os.path.exists(history_path):
            with open(history_path, "r") as file:
                self.chat_history = json.load(file)
            for sender, message, timestamp in self.chat_history:
                self.add_message(sender, message)
        else:
            print("No chat history file found.")

    def create_file_drop_zone(self):
        """Create a file drop zone in the sidebar"""
        drop_zone = ctk.CTkFrame(self.sidebar, fg_color=("gray85", "gray20"), corner_radius=10)
        drop_zone.pack(fill="x", padx=10, pady=10)
        
        # Drop zone label
        label = ctk.CTkLabel(drop_zone, 
                           text="📁 Drop CSV file here\nor click to select",
                           font=ctk.CTkFont(family="Segoe UI", size=12),
                           text_color=("gray30", "gray70"))
        label.pack(padx=20, pady=20)
        
        # Make the frame droppable
        drop_zone.drop_target_register(DND_FILES)
        drop_zone.dnd_bind('<<Drop>>', self.on_drop)
        
        # Bind click event to open file dialog
        drop_zone.bind('<Button-1>', self.on_click_select_file)
        
    def on_drop(self, event):
        """Handle file drop event"""
        # Get the dropped file path
        file_path = event.data.strip('{}')
        
        # Handle multiple files (take the first one)
        if '\n' in file_path:
            file_path = file_path.split('\n')[0]
            
        # Import the data
        success, message = import_data(file_path)
        self.add_message("System", message)
            
    def on_click_select_file(self, event):
        """Handle click event to open file dialog"""
        file_path = ctk.filedialog.askopenfilename(
            title="Select CSV file",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        
        if file_path:
            success, message = import_data(file_path)
            self.add_message("System", message)

    def main(self):
        # Load chat history on startup
        self.load_chat_history()

        # Start the Tkinter event loop
        self.mainloop()

if __name__ == "__main__":
    app = DataScienceCopilotApp()
    app.main()