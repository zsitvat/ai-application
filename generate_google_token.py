#!/usr/bin/env python3
"""
Script to generate Google OAuth2 token for Drive API access.
Run this once to create token.json file.
"""

import os
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow

SCOPES = ["https://www.googleapis.com/auth/drive.file"]


def main():
    creds = None
    token_file = "token.json"
    credentials_file = "credentials.json"

    if not os.path.exists(credentials_file):
        print(f"Error: {credentials_file} not found!")
        return

    if os.path.exists(token_file):
        creds = Credentials.from_authorized_user_file(token_file, SCOPES)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            print("Refreshing expired token...")
            creds.refresh(Request())
        else:
            print("Starting OAuth2 flow...")
            print("Using manual authorization flow...")
            print(
                "Please add http://localhost:8080/ to your OAuth2 redirect URIs in Google Cloud Console"
            )
            print("Or use this manual method:")
            print()

            # Manual authorization flow with OOB redirect
            flow = InstalledAppFlow.from_client_secrets_file(credentials_file, SCOPES)
            flow.redirect_uri = "urn:ietf:wg:oauth:2.0:oob"
            auth_url, _ = flow.authorization_url(prompt="consent")
            print(f"1. Go to this URL: {auth_url}")
            print("2. Sign in and authorize the application")
            print("3. You will see an authorization code on the page")
            auth_code = input("4. Enter the authorization code here: ")
            flow.fetch_token(code=auth_code)
            creds = flow.credentials

        print(f"Saving token to {token_file}")
        with open(token_file, "w") as token:
            token.write(creds.to_json())

    print("✅ Token is valid and ready to use!")


if __name__ == "__main__":
    main()
