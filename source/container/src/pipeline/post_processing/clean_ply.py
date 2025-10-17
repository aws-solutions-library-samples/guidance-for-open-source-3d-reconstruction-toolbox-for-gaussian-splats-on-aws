#!/usr/bin/env python3
"""
Remove comment lines from PLY files to fix SPZ compatibility issues.
"""

import argparse
import sys

def main():
    parser = argparse.ArgumentParser(description='Remove comment lines from PLY file')
    parser.add_argument('-i', '--input', required=True, help='Input PLY file path')
    parser.add_argument('-o', '--output', help='Output PLY file path (default: overwrite input)')
    
    args = parser.parse_args()
    
    output_path = args.output if args.output else args.input
    
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