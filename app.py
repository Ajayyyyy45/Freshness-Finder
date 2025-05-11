import streamlit as st
import streamlit_authenticator as stauth
import yaml
from yaml.loader import SafeLoader

# Load configuration from YAML
with open('config.yaml') as file:
    config = yaml.load(file, Loader=SafeLoader)

# Create authenticator
authenticator = stauth.Authenticate(
    config['credentials'],
    config['cookie']['name'],
    config['cookie']['key'],
    config['cookie']['expiry_days'],
    config['preauthorized']
)

# Login (VALID LOCATIONS: 'main', 'sidebar', 'unrendered')
auth_status = authenticator.login('Login', 'main')  # Change 'main' to 'sidebar' if preferred

# Logic based on login status
if auth_status:
    authenticator.logout('Logout', 'main')
    st.write(f'Welcome *{st.session_state["name"]}*')
    st.title('Some App Content Here')
elif auth_status is False:
    st.error('Username/password is incorrect')
elif auth_status is None:
    st.warning('Please enter your username and password')









       
