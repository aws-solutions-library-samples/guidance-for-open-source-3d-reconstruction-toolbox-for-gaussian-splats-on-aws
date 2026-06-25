#!/usr/bin/env python3
# MIT License
#
# Copyright (c) 2025 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY

"""
Remove comment lines from PLY files to fix SPZ compatibility issues.
"""

import argparse
import os
import sys

def main():
    parser = argparse.ArgumentParser(description='Remove comment lines from PLY file')
    parser.add_argument('-i', '--input', required=True, help='Input PLY file path')
    parser.add_argument('-o', '--output', help='Output PLY file path (default: overwrite input)')
    
    args = parser.parse_args()
    
    output_path = args.output if args.output else args.input

    # Skip gracefully if input file doesn't exist (e.g. export failed or was skipped)
    if not os.path.exists(args.input):
        print(f"Info: PLY file not found, skipping comment removal: {args.input}")
        sys.exit(0)

    try:
        with open(args.input, 'rb') as infile:
            content = infile.read()
        
        # Find the end of header (marked by "end_header\n")
        end_header_marker = b'end_header\n'
        header_end = content.find(end_header_marker)
        
        if header_end == -1:
            print("Error: Could not find end_header marker in PLY file")
            sys.exit(1)
        
        # Split into header and data
        header_bytes = content[:header_end + len(end_header_marker)]
        data_bytes = content[header_end + len(end_header_marker):]
        
        # Process header as text
        try:
            header_text = header_bytes.decode('utf-8')
        except UnicodeDecodeError:
            header_text = header_bytes.decode('latin-1')
        
        header_lines = header_text.splitlines(keepends=True)
        
        # Filter out comment lines from header only
        cleaned_header_lines = []
        for line in header_lines:
            stripped = line.strip()
            # Only remove lines that start with 'comment ' (with space) or are just 'comment'
            if stripped.startswith('comment ') or stripped == 'comment':
                continue
            cleaned_header_lines.append(line)
        
        # Write cleaned header + original binary data
        with open(output_path, 'wb') as outfile:
            cleaned_header = ''.join(cleaned_header_lines).encode('utf-8')
            outfile.write(cleaned_header)
            outfile.write(data_bytes)
        
        print(f"Cleaned PLY file: removed {len(header_lines) - len(cleaned_header_lines)} comment lines from header")
        
    except Exception as e:
        print(f"Error cleaning PLY file: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
