from cryptography.hazmat.primitives import serialization, hashes
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.backends import default_backend
import base64
import os

PRIVATE_KEY_PEM = os.getenv("MAGI_PRIVATE_KEY_PEM").replace('\\n', '\n') #cannot have \n in environment variables, but necessary in the PEM format
PUBLIC_KEY_PEM = os.getenv("MAGI_PUBLIC_KEY_PEM").replace('\\n', '\n')     

def encrypt_value(value):
    if not value:
        return "No value to encrypt"
    try:
        # Load the public key
        public_key_bytes = PUBLIC_KEY_PEM.encode('utf-8')
        public_key = serialization.load_pem_public_key(
            public_key_bytes,
            backend=default_backend()
        )
        
        # Encrypt and encode
        encrypted = public_key.encrypt(
            value.encode(),
            padding.OAEP(
                mgf=padding.MGF1(algorithm=hashes.SHA256()),
                algorithm=hashes.SHA256(),
                label=None
            )
        )
        return base64.b64encode(encrypted).decode()
    except Exception as e:
        print(e)
        return f"Encryption failed: {str(e)}"
    

def decrypt_value(encrypted_value):
    if not encrypted_value:
        return "No encrypted value to decrypt"
    try:
        # Load the private key
        private_key_bytes = PRIVATE_KEY_PEM.encode('utf-8')
        private_key = serialization.load_pem_private_key(
            private_key_bytes,
            password=None,
            backend=default_backend()
        )
        
        # Decode and decrypt
        encrypted_bytes = base64.b64decode(encrypted_value)
        decrypted_bytes = private_key.decrypt(
            encrypted_bytes,
            padding.OAEP(
                mgf=padding.MGF1(algorithm=hashes.SHA256()),
                algorithm=hashes.SHA256(),
                label=None
            )
        )
        return decrypted_bytes.decode()
    except Exception as e:
        print(e)
        return f"Decryption failed: {str(e)}"
    
#HEAD_JS is the function that will be used by the browser to encrypt API keys and store them in local storage
#it is a template with a place-holder for the public key that should be used for encryption
#this key should be retrieved from the server's environment variables at initialization
#changing the private/public key will invalidate any encrypted API keys stored in users' browsers
HEAD_JS = """
<script>
async function encryptValue(value) {
    if (!value) return '';
    
    // Import the public key
    const publicKey = await window.crypto.subtle.importKey(
        'spki',
        str2ab(atob(PUBLIC_KEY_PEM.replace(/-----BEGIN PUBLIC KEY-----/, '')
            .replace(/-----END PUBLIC KEY-----/, '')
            .replace(/\\n/g, ''))),
        {
            name: 'RSA-OAEP',
            hash: 'SHA-256',
        },
        false,
        ['encrypt']
    );

    // Encrypt the value
    const encoded = new TextEncoder().encode(value);
    const encrypted = await window.crypto.subtle.encrypt(
        {
            name: 'RSA-OAEP'
        },
        publicKey,
        encoded
    );

    // Convert to base64 and store
    const base64 = btoa(String.fromCharCode(...new Uint8Array(encrypted)));
    localStorage.setItem('encrypted_value', base64);
    return base64;
}

function str2ab(str) {
    const buf = new ArrayBuffer(str.length);
    const bufView = new Uint8Array(buf);
    for (let i = 0, strLen = str.length; i < strLen; i++) {
        bufView[i] = str.charCodeAt(i);
    }
    return buf;
}

// Make public key available to JavaScript
const PUBLIC_KEY_PEM = {REPLACE_ME};

console.log('public key', PUBLIC_KEY_PEM);

</script>
"""

print(PUBLIC_KEY_PEM)

head_js = HEAD_JS.replace("{REPLACE_ME}", f'`{PUBLIC_KEY_PEM}`')

def generate_keys():

    # Generate RSA key pair (do this once and store keys securely in production)
    private_key = rsa.generate_private_key(
        public_exponent=65537,
        key_size=2048,
        backend=default_backend()
    )
    
    private_key_pem = private_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.TraditionalOpenSSL,
        encryption_algorithm=serialization.NoEncryption()
    )

    # Export public key in PEM format for frontend
    public_key = private_key.public_key()
    public_key_pem = public_key.public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo
    )
    
    private_key_pem_str = private_key_pem.decode("utf-8")
    public_key_pem_str = public_key_pem.decode("utf-8")

    return private_key_pem_str, public_key_pem_str #these will contain \n


if __name__ == "__main__":

    #comment out this line if you want to test your environment variables
    PRIVATE_KEY_PEM, PUBLIC_KEY_PEM = generate_keys()

    print("COPY THE FOLLOWING INTO YOUR ENVIRONMENT VARIABLES")

    print(PRIVATE_KEY_PEM.replace('\n', '\\n')) #cannot have \n in environment variables, but necessary in the PEM format

    print()

    print(PUBLIC_KEY_PEM.replace('\n', '\\n'))

    print()
    print("TESTING ENCRYPTION AND DECRYPTION")

    msg = "hello world"
    encrypted = encrypt_value(msg)
    print(encrypted)
    decrypted = decrypt_value(encrypted)
    print(decrypted)
