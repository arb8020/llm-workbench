#!/usr/bin/env python3
"""Script to remove SSH private keys from git history"""

import re
import sys

# Pattern to match SSH private keys
ssh_key_patterns = [
    r'-----BEGIN OPENSSH PRIVATE KEY-----.*?-----END OPENSSH PRIVATE KEY-----',
    r'-----BEGIN RSA PRIVATE KEY-----.*?-----END RSA PRIVATE KEY-----',
    r'-----BEGIN PRIVATE KEY-----.*?-----END PRIVATE KEY-----',
    r'ssh-rsa AAAA[0-9A-Za-z+/]+[=]{0,3}',
    r'ssh-ed25519 AAAA[0-9A-Za-z+/]+[=]{0,3}',
]

def clean_content(content):
    """Remove SSH keys from content"""
    for pattern in ssh_key_patterns:
        content = re.sub(pattern, '[REDACTED SSH KEY]', content, flags=re.DOTALL | re.MULTILINE)
    return content

if __name__ == '__main__':
    content = sys.stdin.read()
    cleaned = clean_content(content)
    sys.stdout.write(cleaned)