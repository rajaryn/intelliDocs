from datetime import datetime, timedelta
from dotenv import load_dotenv
import os
load_dotenv()

from flask import Flask,session
from flask import render_template, render_template_string
from flask import request,redirect
from flask import flash,url_for
import cloudinary
import cloudinary.uploader
import cloudinary.api
from flask_bcrypt import Bcrypt
import database
import os
import secrets
import processing
import fitz
import ai_utils
import vector_store
import rag
import mongodb
import email_server
import secrets
import hashlib
from user_manager import delete_user_account

database.init_db()  # Ensures DB and tables exist

app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv('FLASK_SECRET_KEY')
bcrypt = Bcrypt(app)


# --- Cloudinary Configuration ---
cloudinary.config(
    cloud_name=os.getenv('CLOUDINARY_CLOUD_NAME'),
    api_key=os.getenv('CLOUDINARY_API_KEY'),
    api_secret=os.getenv('CLOUDINARY_API_SECRET')
)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/signup', methods=['GET','POST'])
@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        confirm_password = request.form['confirm_password']

        # --- Basic Validation ---
        if password != confirm_password:
            error_message = "Passwords do not match."
            return render_template('signup.html', error_message=error_message, email=email)
        
        # Check if user already exists
        existing_user = database.get_user_by_email(email)
        if existing_user:
            flash('An account with this email already exists.', 'danger')
            return render_template('signup.html', email=email)
        
        # Hash the password
        password_hash = bcrypt.generate_password_hash(password).decode('utf-8')

        # Add user to the database
        new_user_id = database.add_user(email, password_hash)

        # Auto Login
        if new_user_id:
            session['user_id'] = new_user_id
            session['user_email'] = email
            
            flash('Account created and you are now logged in!', 'success')
            return redirect(url_for('dashboard')) 
        else:
            flash('An error occurred while creating your account. Please try again.', 'danger')
            return render_template('signup.html', email=email)

    # It handles the GET request (showing the page).
    return render_template('signup.html')
        

@app.route('/login', methods=['GET', 'POST'])
def login():
   
    if request.method == 'POST':
       
        email = request.form['email']
        password = request.form['password']
        user = database.get_user_by_email(email)

        if user and bcrypt.check_password_hash(user['password_hash'], password):
          
            session['user_id'] = user['id']
            session['user_email'] = user['email']
            
            flash('Login successful! Welcome back.', 'success')
            return redirect(url_for('dashboard')) 
        else:
           
            flash('Invalid email or password. Please try again.', 'danger')
            return redirect(url_for('login')) 
        
    
    return render_template('login.html')

        
@app.route('/logout', methods=['POST'])
def logout():
   
    # Clear all data from the session dictionary
    session.clear()
    
    flash('You have been successfully logged out.', 'info')
    
    # Redirect the user to the login page
    return redirect(url_for('login'))

"""Generates a secure, random token for the email link and 
a hash of that token to store in the database."""
def generate_secure_reset_token():
    
    raw_token = secrets.token_urlsafe(32)
    
    # Generate the hash of that token
    token_hash = hashlib.sha256(raw_token.encode('utf-8')).hexdigest()
    
    return raw_token, token_hash

@app.route("/forgot_password", methods=['GET', 'POST'])
def forgot_password():
    
    if request.method == 'POST':
        
        email = request.form.get('email')
        user=database.get_user_by_email(email)
        
        if user:

            token, token_hash = generate_secure_reset_token()
            expires_at = datetime.now() + timedelta(minutes=15)

            stored_successfully = database.store_reset_token(user['id'], token_hash, expires_at)

            if stored_successfully:
             reset_link = url_for('reset_password', token=token, _external=True)
             email_server.send_reset_email(email, reset_link)

            else:
             flash("An error occurred. Please try again.", "error")
             return redirect(url_for('forgot_password'))

        return render_template_string("""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Reset Sent</title>
        <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/@picocss/pico@2/css/pico.min.css" />
        <style>
            /* Custom styles to match the login/reset forms */
            body { 
                display: flex; 
                justify-content: center; 
                align-items: center; 
                min-height: 100vh;
                background-color: var(--pico-background-color);
                font-family: var(--pico-font-family);
            }
            /* --- FIX: Center the message-card horizontally within the container --- */
            main.container { 
                max-width: 500px; 
                /* Set container width to auto and use margin: 0 auto for centering */
                width: 100%;
                margin: 0 auto; 
            }
            .message-card {
                max-width: 500px; 
                padding: 2rem; 
                background-color: var(--pico-color-bg); 
                border: 1px solid var(--pico-color-border);
                border-radius: var(--pico-border-radius);
                box-shadow: var(--pico-box-shadow);
            }
            .success-text {
                color: var(--pico-color-green-600);
            }
        </style>
    </head>
    <body>
        <main class="container">
            <div class="message-card">
                <h3 style='text-align: center; padding: 2rem;' class="success-text">
                    If an account with that email exists, a reset link has been sent.
                </h3>
            </div>
        </main>
    </body>
    </html>
""")
    return render_template("forgot_password.html")

@app.route("/reset-password", methods=['GET', 'POST'])
def reset_password():
    
    if request.method == 'POST':
        # --- POST: Handling the Form Submission and Password Update ---
        
        # 1. Get data from the form
        token = request.form.get('token')
        new_password = request.form.get('password')
        confirm_password = request.form.get('password_confirm')

        # Basic Validation
        if new_password != confirm_password:
            flash("Passwords do not match.", "error")
            # Re-rendering the form, passing the token back to avoid losing state
            return render_template("reset_password.html", token=token)

        # 2. Secure Token Validation
        token_hash = hashlib.sha256(token.encode('utf-8')).hexdigest()
        reset_request = database.get_reset_token_details(token_hash)
        
        # Check if the token exists AND is not expired
        if not reset_request or reset_request['expires_at'] < datetime.now():
            flash("This reset link is invalid or has expired.", "error")
            return redirect(url_for('forgot_password'))

        # 3. Token is Valid: Perform the Update
        user_id = reset_request['user_id']
        
        # Securely hash the new password using Bcrypt
        hashed_password = bcrypt.generate_password_hash(new_password).decode('utf-8')
        
        database.update_user_password(user_id, hashed_password)
        
        database.delete_reset_token(token_hash)
        
        flash("Your password has been updated! You can now log in.", "success")
        return redirect(url_for('login'))


    # --- GET: Displaying the Form (User clicked the link) ---
    token = request.args.get('token')

    # 1. Checking for token existence in the URL
    if not token:
        flash("Invalid reset link. No token provided.", "error")
        return redirect(url_for('forgot_password'))

    # 2. Validating the token before showing the form
    token_hash = hashlib.sha256(token.encode('utf-8')).hexdigest()
    reset_request = database.get_reset_token_details(token_hash)

    # Checking validity (token exists AND is not expired)
    if not reset_request or reset_request['expires_at'] < datetime.now():
        flash("The password reset link is invalid or has expired.", "error")
        return redirect(url_for('forgot_password'))

    # 3. Token is valid: Show the password form
    # We pass the token so it can be saved in a hidden field for the POST request
    return render_template("reset_password_form.html", token=token)

@app.route('/dashboard')
def dashboard():
    # Check if user is logged in
    if 'user_id' not in session:
        flash('Please log in to access this page.', 'warning')
        return redirect(url_for('login'))

   
    # Fetch the documents for the currently logged-in user.
    user_id = session['user_id']
    
    if 'user_id' not in session:
        flash('Please log in to access this page.', 'warning')
        return redirect(url_for('login'))
    
    documents = database.get_documents_by_user(user_id)
    documents_for_template=[]
    for doc in documents:
        tags=doc["tags"]
        tags_list = [tag.strip() for tag in tags.split(',')]

        clean_document = {
        'id': doc["id"],
        'filename': doc["filename"],
        'url':doc["url"],
        'tags': tags_list,
        'created_at': doc["created_at"]
        }
        print(clean_document)
        documents_for_template.append(clean_document)
    
    return render_template('dashboard.html', documents=documents_for_template)

# Helper function to check for allowed file types
ALLOWED_EXTENSIONS = {'pdf'}
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def generate_unique_public_id(original_filename):
    """
    Takes a filename, sanitizes it, and adds a unique suffix.
    Returns a string suitable for a Cloudinary public_id.
    """
    # 1. Get the original filename and separate its name and extension
    filename_without_ext, file_ext = os.path.splitext(original_filename)
    
    # 2. Sanitize the base filename (replace non-alphanumeric chars with '_')
    sanitized_filename = "".join(c if c.isalnum() else "_" for c in filename_without_ext)
    
    # 3. Create a unique suffix using 4 random bytes (8 hex characters)
    unique_suffix = secrets.token_hex(4)
    
    # 4. Construct and return the final, unique public_id
    return f"{sanitized_filename}_{unique_suffix}"

@app.route('/upload', methods=['POST'])
def upload_document():
    if 'user_id' not in session:
        flash('Please log in to upload files.', 'danger')
        return redirect(url_for('login'))

   
    if 'file' not in request.files:
        flash('No file part in the request.', 'danger')
        return redirect(url_for('dashboard'))

    file = request.files['file']

    # 3. Check if the user selected a file
    if file.filename == '':
        flash('No file selected.', 'warning')
        return redirect(url_for('dashboard'))

    # 4. Validate the file type
    if file and allowed_file(file.filename):
        try:
           
            file_bytes=file.read()
          
            final_public_id = generate_unique_public_id(file.filename)

            # --- AI LOGIC ---
            text_content = ""
            with fitz.open(stream=file_bytes, filetype="pdf") as doc:
                for page in doc:
                    text_content += page.get_text()
            
            tags_list = ai_utils.generate_tags_for_text(text_content)
            tags_string = ",".join(tags_list)

            summary = ai_utils.generate_summary_for_text(text_content)

            # --- Rewind the file stream back to the beginning IMP---
            file.seek(0)

            # Upload the file to Cloudinary
            # 'raw' because it's a non-image file (PDF)
            upload_result = cloudinary.uploader.upload(
                file, 
                public_id=final_public_id,
                resource_type='raw',
                )
        
            
            url = upload_result.get('secure_url')
            public_id = upload_result.get('public_id')
            user_id = session['user_id']
            
            new_doc_id=database.add_document(user_id, file.filename, url, public_id, tags_string,summary)

            if new_doc_id:
                database.update_document_status(new_doc_id, 'PROCESSING')
                processing.process_and_index_pdf(new_doc_id, file_bytes)

                flash('File uploaded successfully! Processing for search has begun.', 'success')
            else:
                flash('Failed to save file information to the database.', 'danger')

        except Exception as e:
            flash(f'An error occurred during upload: {e}', 'danger')
            
        return redirect(url_for('dashboard'))
    else:
        flash('Invalid file type. Only PDF files are allowed.', 'danger')
        return redirect(url_for('dashboard'))

@app.route('/delete/<int:doc_id>', methods=['POST'])
def delete_document(doc_id):
    
    # 1. Check if user is logged in
    if 'user_id' not in session:
        flash('You must be logged in to delete files.', 'danger')
        return redirect(url_for('login'))
    
    # 2. Fetch the document's metadata from our database
    document_to_delete = database.get_document_by_id(doc_id)
    
    # 3. Check if the document exists
    if not document_to_delete:
        flash('Document not found or it may have already been deleted.', 'warning')
        return redirect(url_for('dashboard'))

    # 4. CRUCIAL SECURITY CHECK: Verify the logged-in user owns this document
    if document_to_delete['user_id'] != session['user_id']:
        flash('You are not authorized to delete this document.', 'danger')
        return redirect(url_for('dashboard'))
    
    print("DEBUG: Security check passed. Proceeding with deletion.")
    try:
        # 5. If all checks pass, delete the file from Cloudinary
        public_id = document_to_delete['public_id']
        cloudinary.uploader.destroy(public_id, resource_type='raw')
        
        # 6. If Cloudinary deletion is successful, delete the record from our database
        if database.delete_document_record(doc_id):
            flash('Document deleted successfully.', 'success')
        else:
            flash('File was deleted from storage, but failed to be removed from the database.', 'danger')

    except Exception as e:
        flash(f'An error occurred while deleting the file: {e}', 'danger')
    
    # 7. Finally, redirect back to the dashboard
    return redirect(url_for('dashboard'))

@app.route('/view/<int:doc_id>')
def view_document(doc_id):
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    document = database.get_document_by_id(doc_id)
    
    if not document or document['user_id'] != session['user_id']:
        flash('Document not found or you are not authorized to view it.', 'danger')
        return redirect(url_for('dashboard'))
    
    
   
    return render_template(
        'view_document.html', 
        document=document,
        document_url=document['url'], 
        document_filename=document['filename'],
        document_summary=document['summary'],
        search_query="",     
        search_results=[],    
        search_error=None     
    )

@app.route('/document/<int:doc_id>/search', methods=['POST'])
def search_in_document(doc_id):
    
    # 1. Authentication and Authorization
    if 'user_id' not in session:
        flash('Please log in to search.', 'danger')
        return redirect(url_for('login'))

    document = database.get_document_by_id(doc_id)
    
    if not document or document['user_id'] != session['user_id']:
        flash('Document not found or you are not authorized.', 'danger')
        return redirect(url_for('dashboard'))

    # 2. Get form data
    query = request.form.get("query")
    search_results = []
    search_error = None

    # 3. Call your vector store
    if not query:
        search_error = "Please enter a search query."
    else:
        try:
            search_results = vector_store.search_document(
                doc_id=doc_id,
                query_text=query,
                top_k=3  # Get the top 3 results
            )
            if not search_results:
                search_error = "No relevant results found."
        except Exception as e:
            print(f"Search error for doc {doc_id}: {e}") # Log the error
            search_error = "An error occurred during search."

    # 4. Re-render the same page, but pass in the search data
    return render_template(
        'view_document.html',
        document=document,
        document_url=document['url'],
        document_filename=document['filename'],
        document_summary=document['summary'],
        search_query=query,       
        search_results=search_results,
        search_error=search_error    
    )


@app.route('/chat/<int:doc_id>', methods=['POST'])
def chat_with_document(doc_id):
    
    # 1. Authentication and Authorization
    if 'user_id' not in session:
        # User not logged in
        return {"error": "Unauthorized. Please log in."}, 401

    document = database.get_document_by_id(doc_id)
    
    if not document or document['user_id'] != session['user_id']:
       return {"error": "Document not found or access denied."}, 404

    # 2. Get the user's message from the JSON body
    data = request.get_json()
    message = data.get("message")

    if not message:
        return {"error": "No message provided."}, 400

    try:

        mongodb.save_message_to_history(str(doc_id), "user", message)
        ai_reply = rag.answer_from_document(doc_id,message)

        mongodb.save_message_to_history(str(doc_id), "assistant", ai_reply)
        
        # 4. Return the AI's response
        return {"reply": ai_reply}

    except Exception as e:
        print(f"Error in chat endpoint for doc {doc_id}: {e}")
        return {"error": f"An internal server error occurred: {str(e)}"}, 500


@app.route('/delete_account', methods=['POST'])
def delete_account():
    # 1. Security Check: Ensure user is actually logged in
    if 'user_id' not in session:
        flash("You must be logged in to perform this action.", "danger")
        return redirect(url_for('login')) # Adjust 'login' to whatever your login route function is called
    
    user_id = session['user_id']
    
    print(f"User {user_id} requested account deletion.")

    # 2. Call the Manager Function
    success, message = delete_user_account(user_id)
    
    # 3. Handle Result
    if success:
        # A. Cleanup the session (Log them out)
        session.clear()
        
        # B. User Feedback
        flash("Your account and all associated data (documents, chat history) have been permanently deleted.", "success")
        
        # C. Redirect to Home/Landing page
        return redirect(url_for('index')) # Adjust 'index' to your home route
        
    else:
        # If something went wrong (e.g., Database error)
        flash(f"Error deleting account: {message}", "danger")
        return redirect(url_for('dashboard')) # Redirect back to their dashboard/settings


if __name__ == "__main__":
    app.run(debug=True,use_reloader=True)
