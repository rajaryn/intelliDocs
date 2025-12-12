import os,dotenv
import certifi
import mysql.connector
from mysql.connector import Error,pooling


# function to be called from app.py after load_dotenv()
def create_db_pool():
    """Creates a connection pool for the database."""
    try:
        db_config = {
            "pool_name": "intellidocs_pool",
            "pool_size": 5,
            "host": os.getenv('DB_HOST', 'localhost'),
            "user": os.getenv('DB_USER', 'root'),
            "password": os.getenv('DB_PASSWORD'),
            "database": os.getenv('DB_NAME', 'notes_db'),
            "port": int(os.getenv('DB_PORT', 3306)), # Added Port
            "autocommit": True
        }

        # Check if we are in "Cloud Mode" (if SSL is needed)
        if os.getenv('DB_SSL') == 'True':
            print("Enabling SSL for Cloud Database...")
            db_config["ssl_ca"] = "/etc/ssl/certs/ca-certificates.crt" # Linux/Mac default
            db_config["ssl_ca"] = certifi.where() 
            db_config["ssl_verify_identity"] = True

        pool = pooling.MySQLConnectionPool(**db_config)
        
        print("Database connection pool created successfully.")
        return pool
    except Error as e:
        print(f"Error creating connection pool: {e}")
        return None
    
cnx_pool = create_db_pool()

def get_db_connection():
    """Gets a connection from the pool."""
    if cnx_pool is None:
        print("Connection pool is not available.")
        return None
    try:
        # Get a connection from the pool
        conn = cnx_pool.get_connection()
        return conn
    except Error as e:
        print(f"Error getting connection from pool: {e}")
        return None

def init_db():
    # --- CHECK: CLOUD VS LOCAL ---
    if os.getenv('DB_SSL') == 'True':
        print("Cloud Mode: Skipping database creation (assuming pre-created).")
    else:
        # Step 1: Create Database (Local Only)
        conn = None
        cursor = None
        try:
            print("Connecting to MySQL server to ensure database exists...")
            conn = mysql.connector.connect(
                host=os.getenv('DB_HOST', 'localhost'),
                user=os.getenv('DB_USER', 'root'),
                password=os.getenv('DB_PASSWORD')
            )
            cursor = conn.cursor()
            db_name = os.getenv('DB_NAME', 'notes_db')
            cursor.execute(f"CREATE DATABASE IF NOT EXISTS {db_name} DEFAULT CHARACTER SET 'utf8mb4'")
            print(f"Database '{db_name}' is ready.")

        except Error as e:
            print(f"Error during initial database creation: {e}")
        finally:
            if conn and conn.is_connected():
                cursor.close()
                conn.close()

    # --- STEP 2: CREATE TABLES (Runs for BOTH Cloud and Local) ---
    # Note: This is now un-indented to match the 'if/else' level
    conn = get_db_connection()
    if conn is None:
        print("Could not get DB connection from pool to create tables.")
        return
    
    try:
        cursor = conn.cursor()
        print("Ensuring tables are created...")
        
        users_table_sql = """
        CREATE TABLE IF NOT EXISTS users (
            id INT AUTO_INCREMENT PRIMARY KEY,
            email VARCHAR(255) UNIQUE NOT NULL,
            password_hash VARCHAR(255) NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        ) ENGINE=InnoDB;
        """
        cursor.execute(users_table_sql)
        print("'users' table is ready.")

        documents_table_sql = """ 
        CREATE TABLE IF NOT EXISTS documents (
            id INT AUTO_INCREMENT PRIMARY KEY,
            user_id INT NOT NULL,
            filename VARCHAR(255) NOT NULL,
            url VARCHAR(512) NOT NULL,
            public_id VARCHAR(255) NOT NULL,
            processing_status ENUM('PENDING', 'PROCESSING', 'COMPLETED', 'FAILED') NOT NULL DEFAULT 'PENDING',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            tags VARCHAR(512) DEFAULT NULL,
            summary TEXT DEFAULT NULL,
            FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
        ) ENGINE=InnoDB;
        """
        cursor.execute(documents_table_sql)
        print("'documents' table is ready.")

        password_resets_table_sql = """ 
        CREATE TABLE IF NOT EXISTS password_resets (
            id INT AUTO_INCREMENT PRIMARY KEY,
            user_id INT NOT NULL,
            token_hash VARCHAR(64) NOT NULL UNIQUE,
            expires_at DATETIME NOT NULL,
            FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
        );
        """
        cursor.execute(password_resets_table_sql)
        print("'password_resets' table is ready.")

    except Error as e:
        print(f"Error during table creation: {e}")
    finally:
        if conn and conn.is_connected():
            cursor.close()
            conn.close()
            print("Connection returned to pool.")

# -------To add a user ---------
def add_user(email, password_hash):
    """Adds a new user to the users table. Returns True on success, False on failure."""
    conn = get_db_connection()
    if conn is None:
        return False
    
    try:
        cursor = conn.cursor()
        # IMPORTANT: Use parameterized queries to prevent SQL injection
        sql = "INSERT INTO users (email, password_hash) VALUES (%s, %s)"
        cursor.execute(sql, (email, password_hash))
        conn.commit()
        
        # Get the ID of the row that was just inserted
        new_user_id = cursor.lastrowid 
        return new_user_id
    
    except Error as e:
        print(f"Error adding user: {e}")
        return False
    finally:
        if conn.is_connected():
            cursor.close()
            conn.close()

# --- To retrieve a user by their email ---
def get_user_by_email(email):

    """Fetches a single user by email. Returns user data as a dict, or None if not found."""
    conn = get_db_connection()
    if conn is None: return None
    try:
        # Use a dictionary cursor to get results as dicts instead of tuples
        cursor = conn.cursor(dictionary=True)
        sql = "SELECT * FROM users WHERE email = %s"
        cursor.execute(sql, (email,))
        user = cursor.fetchone()
        return user
    except Error as e:
        print(f"Error fetching user: {e}")
        return None
    finally:
        if conn.is_connected():
            cursor.close()
            conn.close()


# ---To add a document's metadata ---
def add_document(user_id, filename, url, public_id,tags_string,summary):
    """Adds a new document record to the database. Returns True on success."""
    conn = get_db_connection()
    if conn is None: return False
    try:
        cursor = conn.cursor()
        sql = """
            INSERT INTO documents (user_id, filename, url, public_id,tags,summary)
            VALUES (%s, %s, %s, %s,%s,%s)
        """
        cursor.execute(sql, (user_id, filename, url, public_id,tags_string,summary))
        conn.commit()
        new_doc_id = cursor.lastrowid 
        return new_doc_id
    except Error as e:
        print(f"Error adding document: {e}")
        conn.rollback()
        return None
    finally:
        if conn.is_connected():
            cursor.close()
            conn.close()

def get_documents_by_user(user_id):
    """Fetches all documents for a specific user, ordered by most recent."""
    conn = get_db_connection()
    if conn is None: return []
    try:
        # Using dictionary=True makes the cursor return rows as dictionaries
        cursor = conn.cursor(dictionary=True)
        
        # The SQL query to select documents for a specific user
        # 'WHERE user_id = %s' is the crucial part for security and correctness
        # 'ORDER BY created_at DESC' shows the newest documents first
        sql = "SELECT id, filename, url, created_at, tags FROM documents WHERE user_id = %s ORDER BY created_at DESC"
        
        cursor.execute(sql, (user_id,))
        
        # fetchall() gets all the rows that match the query
        documents = cursor.fetchall()
        return documents
    except Error as e:
        print(f"Error fetching documents: {e}")
        return [] # Return an empty list if an error occurs
    finally:
        if conn.is_connected():
            cursor.close()
            conn.close()

def get_document_by_id(doc_id):
    """Fetches a single document by its primary key ID."""
    conn = get_db_connection()
    if conn is None: return None
    try:
        cursor = conn.cursor(dictionary=True)
        sql = "SELECT * FROM documents WHERE id = %s"
        cursor.execute(sql, (doc_id,))
        document = cursor.fetchone()
        return document
    except Error as e:
        print(f"Error fetching document by id: {e}")
        return None
    finally:
        if conn.is_connected():
            cursor.close()
            conn.close()

def delete_document_record(doc_id):

    """Deletes a document record from the database by its ID."""
    conn = get_db_connection()
    if conn is None: return False
    try:
        cursor = conn.cursor()
        sql = "DELETE FROM documents WHERE id = %s"
        cursor.execute(sql, (doc_id,))
        conn.commit()
        return True
    except Error as e:
        print(f"Error deleting document record: {e}")
        conn.rollback()
        return False
    finally:
        if conn.is_connected():
            cursor.close()
            conn.close()

def update_document_status(doc_id: int, status: str):
    """Updates the processing_status for a specific document."""
    conn = get_db_connection()
    if conn is None: return False
    try:
        # A list of valid statuses to prevent incorrect values
        valid_statuses = ['PENDING', 'PROCESSING', 'COMPLETED', 'FAILED']
        if status not in valid_statuses:
            print(f"Invalid status provided: {status}")
            return False

        cursor = conn.cursor()
        sql = "UPDATE documents SET processing_status = %s WHERE id = %s"
        cursor.execute(sql, (status, doc_id))
        conn.commit()
        return True
    except Error as e:
        print(f"Error updating document status: {e}")
        conn.rollback()
        return False
    finally:
        if conn.is_connected():
            cursor.close()
            conn.close()


# --- To store a new password reset token ---
def store_reset_token(user_id, token_hash, expires_at):
    """Saves the hashed reset token to the password_resets table."""
    conn = get_db_connection()
    if conn is None: return False
    try:
        cursor = conn.cursor()
        
        # Clean up any old, expired tokens for this user first
        cursor.execute("DELETE FROM password_resets WHERE user_id = %s", (user_id,))
        
        # Insert the new token
        sql = "INSERT INTO password_resets (user_id, token_hash, expires_at) VALUES (%s, %s, %s)"
        cursor.execute(sql, (user_id, token_hash, expires_at))
        conn.commit()
        return True
    except Error as e:
        print(f"Error storing reset token: {e}")
        conn.rollback()
        return False
    finally:
        if conn.is_connected():
            cursor.close()
            conn.close()

# --- To retrieve a token's details ---
def get_reset_token_details(token_hash):
    """Fetches reset token details by the token_hash. Returns a dict or None."""
    conn = get_db_connection()
    if conn is None: return None
    try:
        cursor = conn.cursor(dictionary=True)
        sql = "SELECT * FROM password_resets WHERE token_hash = %s"
        cursor.execute(sql, (token_hash,))
        token_data = cursor.fetchone()
        return token_data
    except Error as e:
        print(f"Error fetching reset token: {e}")
        return None
    finally:
        if conn.is_connected():
            cursor.close()
            conn.close()

# --- To update a user's password ---
def update_user_password(user_id, new_hashed_password):
    """Updates the user's password_hash in the users table."""
    conn = get_db_connection()
    if conn is None: return False
    try:
        cursor = conn.cursor()
        sql = "UPDATE users SET password_hash = %s WHERE id = %s"
        cursor.execute(sql, (new_hashed_password, user_id))
        conn.commit()
        return True
    except Error as e:
        print(f"Error updating user password: {e}")
        conn.rollback()
        return False
    finally:
        if conn.is_connected():
            cursor.close()
            conn.close()

# --- To delete a token after use ---
def delete_reset_token(token_hash):
    """Deletes a password reset token from the table after it has been used."""
    conn = get_db_connection()
    if conn is None: return False
    try:
        cursor = conn.cursor()
        sql = "DELETE FROM password_resets WHERE token_hash = %s"
        cursor.execute(sql, (token_hash,))
        conn.commit()
        return True
    except Error as e:
        print(f"Error deleting reset token: {e}")
        conn.rollback()
        return False
    finally:
        if conn.is_connected():
            cursor.close()
            conn.close()

