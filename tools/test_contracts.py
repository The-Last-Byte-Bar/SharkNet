# tools/test_contracts.py
from typing import Dict, Optional
import re

def basic_ergoscript_validation(code: str) -> Dict[str, bool]:
    """
    Performs basic validation of ErgoScript code.
    
    Args:
        code: ErgoScript code string
        
    Returns:
        Dict of validation results
    """
    results = {
        'has_valid_brackets': False,
        'has_valid_syntax': False,
        'has_basic_structure': False
    }
    
    # Check bracket matching
    brackets = {'(': ')', '{': '}', '[': ']'}
    stack = []
    
    for char in code:
        if char in brackets.keys():
            stack.append(char)
        elif char in brackets.values():
            if not stack:
                return results
            if char != brackets[stack.pop()]:
                return results
    
    results['has_valid_brackets'] = len(stack) == 0
    
    # Basic syntax check
    basic_syntax = re.compile(r'{.*}', re.DOTALL)
    results['has_valid_syntax'] = bool(basic_syntax.match(code))
    
    # Check basic structure
    required_elements = [
        r'\{',        # Opening brace
        r'\}',        # Closing brace
        r'[^}]+',     # Some content between braces
    ]
    
    pattern = ''.join(required_elements)
    results['has_basic_structure'] = bool(re.match(pattern, code, re.DOTALL))
    
    return results