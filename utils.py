"""Utility functions for log processing and chunking."""

import os
import re
from typing import List, Tuple


def read_log_file(file_path: str) -> str:
    """
    Read a log file and return its content.
    
    Args:
        file_path: Path to the log file
        
    Returns:
        Content of the log file
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Log file not found: {file_path}")
    
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        return f.read()


def chunk_logs(log_content: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    """
    Split log content into overlapping chunks for better context preservation.
    
    Args:
        log_content: Full log content
        chunk_size: Number of characters per chunk
        overlap: Number of overlapping characters between chunks
        
    Returns:
        List of log chunks
    """
    chunks = []
    lines = log_content.split('\n')
    
    current_chunk = []
    current_length = 0
    
    for line in lines:
        line_length = len(line) + 1  # +1 for newline
        
        if current_length + line_length > chunk_size and current_chunk:
            # Save current chunk and create overlap
            chunk_text = '\n'.join(current_chunk)
            chunks.append(chunk_text)
            
            # Create overlap by keeping last few lines
            overlap_lines = int(overlap / 20)  # Rough estimate
            current_chunk = current_chunk[-overlap_lines:] if overlap_lines > 0 else []
            current_length = sum(len(line) + 1 for line in current_chunk)
        
        current_chunk.append(line)
        current_length += line_length
    
    # Add remaining chunk
    if current_chunk:
        chunk_text = '\n'.join(current_chunk)
        chunks.append(chunk_text)
    
    return [chunk for chunk in chunks if chunk.strip()]


def extract_error_patterns(log_content: str) -> List[Tuple[str, List[str]]]:
    """
    Extract common error patterns from logs.
    
    Args:
        log_content: Full log content
        
    Returns:
        List of (pattern_name, matches) tuples
    """
    patterns = {
        'Errors': r'(?i)(error|exception|failed|failure)',
        'Warnings': r'(?i)(warning|warn)',
        'Stack Traces': r'^\s+at\s+|Traceback|File ".*", line',
        'HTTP Errors': r'(4\d{2}|5\d{2})\s',
        'Connection Issues': r'(?i)(connection|timeout|refused|unreachable)',
    }
    
    results = []
    for pattern_name, pattern in patterns.items():
        matches = re.findall(pattern, log_content, re.MULTILINE)
        if matches:
            results.append((pattern_name, list(set(matches))))
    
    return results


def summarize_logs(log_content: str, max_lines: int = 20) -> str:
    """
    Create a summary of logs by extracting key information.
    
    Args:
        log_content: Full log content
        max_lines: Maximum lines to include in summary
        
    Returns:
        Summarized log content
    """
    lines = log_content.split('\n')
    
    # Extract first few lines and error lines
    key_lines = []
    key_lines.extend(lines[:5])  # First 5 lines
    
    # Add error/warning lines
    for i, line in enumerate(lines):
        if 'error' in line.lower() or 'warning' in line.lower() or 'exception' in line.lower():
            # Include context (previous and next line)
            if i > 0:
                key_lines.append(lines[i-1])
            key_lines.append(line)
            if i < len(lines) - 1:
                key_lines.append(lines[i+1])
    
    # Keep last few lines
    key_lines.extend(lines[-5:])
    
    # Remove duplicates while preserving order
    seen = set()
    unique_lines = []
    for line in key_lines[:max_lines]:
        if line not in seen:
            unique_lines.append(line)
            seen.add(line)
    
    return '\n'.join(unique_lines)
