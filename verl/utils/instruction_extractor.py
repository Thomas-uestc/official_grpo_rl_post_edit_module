# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Instruction extraction utilities for image editing tasks
"""

import json
import re
from typing import Dict, List, Optional, Any
import logging

logger = logging.getLogger(__name__)


class InstructionExtractor:
    """Extract re-edit instructions from Qwen2.5-VL-7B model outputs"""
    
    def __init__(self, fallback_instruction: str = "Improve the image quality and consistency"):
        """
        Initialize the instruction extractor
        
        Args:
            fallback_instruction: Default instruction to use when extraction fails
        """
        self.fallback_instruction = fallback_instruction
    
    def extract_re_edit_instruction(self, model_output: str) -> str:
        """
        Extract the Re_Edit instruction from model output
        
        Args:
            model_output: Raw text output from Qwen2.5-VL-7B model
            
        Returns:
            Extracted re-edit instruction or fallback instruction
        """
        try:
            # Method 1: Try to parse as JSON
            json_instruction = self._extract_from_json(model_output)
            if json_instruction:
                return json_instruction
            
            # Method 2: Try to extract using regex patterns
            regex_instruction = self._extract_with_regex(model_output)
            if regex_instruction:
                return regex_instruction
            
            # Method 3: Try to extract from structured text
            structured_instruction = self._extract_from_structured_text(model_output)
            if structured_instruction:
                return structured_instruction
            
            logger.warning(f"Failed to extract instruction from: {model_output[:200]}...")
            return self.fallback_instruction
            
        except Exception as e:
            logger.error(f"Error extracting instruction: {e}")
            return self.fallback_instruction
    
    def _extract_from_json(self, text: str) -> Optional[str]:
        """Extract Re_Edit from JSON format"""
        try:
            # Try to find JSON block in the text
            json_patterns = [
                r'\{[^{}]*"Re_Edit"[^{}]*\}',  # Simple JSON object
                r'\{.*?"Re_Edit".*?\}',        # JSON with nested content
            ]
            
            for pattern in json_patterns:
                matches = re.findall(pattern, text, re.DOTALL | re.IGNORECASE)
                for match in matches:
                    try:
                        data = json.loads(match)
                        if isinstance(data, dict) and "Re_Edit" in data:
                            instruction = data["Re_Edit"]
                            if instruction and isinstance(instruction, str):
                                return instruction.strip()
                    except json.JSONDecodeError:
                        continue
            
            # Try to parse the entire text as JSON
            try:
                # Use object_pairs_hook to handle duplicate keys (take first occurrence)
                def handle_duplicates(pairs):
                    result = {}
                    for key, value in pairs:
                        if key not in result:  # Only add if key doesn't exist
                            result[key] = value
                    return result
                
                data = json.loads(text.strip(), object_pairs_hook=handle_duplicates)
                if isinstance(data, dict) and "Re_Edit" in data:
                    instruction = data["Re_Edit"]
                    if instruction and isinstance(instruction, str):
                        return instruction.strip()
            except json.JSONDecodeError:
                pass
                
        except Exception as e:
            logger.debug(f"JSON extraction failed: {e}")
        
        return None
    
    def _extract_with_regex(self, text: str) -> Optional[str]:
        """Extract Re_Edit using regex patterns"""
        try:
            # Pattern 1: XML/HTML tags - <Re_edit>content</Re_edit> (single line and multi-line)
            xml_pattern = r'<Re_edit[^>]*>(.*?)</Re_edit>'
            match = re.search(xml_pattern, text, re.IGNORECASE | re.DOTALL)
            if match:
                instruction = match.group(1).strip()
                # Clean up newlines and extra spaces
                instruction = re.sub(r'\s+', ' ', instruction)
                return instruction.strip()
            
            # Pattern 2: "Re_Edit": "instruction" (JSON format)
            json_pattern = r'"Re_Edit"\s*:\s*"([^"]+)"'
            match = re.search(json_pattern, text, re.IGNORECASE)
            if match:
                return match.group(1).strip()
            
            # Pattern 3: Re_Edit: instruction (without quotes, but avoid XML tags)
            colon_pattern = r'Re_Edit\s*:\s*([^\n\r,}<]+?)(?=\n|$|,|\}|<)'
            match = re.search(colon_pattern, text, re.IGNORECASE)
            if match:
                instruction = match.group(1).strip()
                # Clean up common artifacts
                instruction = re.sub(r'^["\']|["\']$', '', instruction)
                return instruction.strip()
            
            # Pattern 4: Look for instruction after "Re_Edit" (fallback, avoid XML)
            fallback_pattern = r'Re_Edit[:\s]+([^\n\r}<]+?)(?:\n|$|,|\}|<)'
            match = re.search(fallback_pattern, text, re.IGNORECASE | re.MULTILINE)
            if match:
                instruction = match.group(1).strip()
                # Clean up common artifacts
                instruction = re.sub(r'^["\']|["\']$', '', instruction)
                return instruction.strip()
                
        except Exception as e:
            logger.debug(f"Regex extraction failed: {e}")
        
        return None
    
    def _extract_from_structured_text(self, text: str) -> Optional[str]:
        """Extract instruction from structured text format"""
        try:
            # First try XML tag extraction for multi-line format
            xml_multiline_pattern = r'<Re_edit[^>]*>\s*\n\s*(.*?)\s*\n\s*</Re_edit>'
            match = re.search(xml_multiline_pattern, text, re.IGNORECASE | re.DOTALL)
            if match:
                instruction = match.group(1).strip()
                # Clean up extra whitespace
                instruction = re.sub(r'\s+', ' ', instruction)
                return instruction.strip()
            
            lines = text.split('\n')
            
            # Look for lines containing "Re_Edit" or similar
            for i, line in enumerate(lines):
                line_lower = line.lower()
                if 're_edit' in line_lower or 're-edit' in line_lower or 'reedit' in line_lower:
                    # Check current line for XML tags
                    xml_match = re.search(r'<re_edit[^>]*>(.*?)</re_edit>', line, re.IGNORECASE)
                    if xml_match:
                        instruction = xml_match.group(1).strip()
                        return instruction if instruction else None
                    
                    # Check current line for other formats
                    instruction = self._extract_from_line(line)
                    if instruction:
                        return instruction
                    
                    # Check next line
                    if i + 1 < len(lines):
                        instruction = self._extract_from_line(lines[i + 1])
                        if instruction:
                            return instruction
            
            # Look for instruction-like patterns as fallback
            instruction_patterns = [
                r'(?:improve|enhance|adjust|modify|edit|change|fix|correct|make|add|remove)\s+[^.!?]+[.!?]?',
                r'(?:make|add|remove|change|adjust)\s+[^.!?]+[.!?]?',
            ]
            
            for pattern in instruction_patterns:
                matches = re.findall(pattern, text, re.IGNORECASE)
                if matches:
                    # Return the longest match
                    return max(matches, key=len).strip()
                    
        except Exception as e:
            logger.debug(f"Structured text extraction failed: {e}")
        
        return None
    
    def _extract_from_line(self, line: str) -> Optional[str]:
        """Extract instruction from a single line"""
        try:
            # Remove common prefixes
            line = re.sub(r'^(Re_Edit|Re-Edit|Reedit)[:\s]*', '', line, flags=re.IGNORECASE)
            line = line.strip()
            
            # Remove quotes
            line = re.sub(r'^["\']|["\']$', '', line)
            line = line.strip()
            
            # Check if it looks like an instruction
            if len(line) > 10 and any(word in line.lower() for word in ['improve', 'enhance', 'adjust', 'modify', 'edit', 'change', 'fix', 'correct', 'make', 'add', 'remove']):
                return line
                
        except Exception as e:
            logger.debug(f"Line extraction failed: {e}")
        
        return None
    
    def extract_batch_instructions(self, model_outputs: List[str]) -> List[str]:
        """
        Extract re-edit instructions from a batch of model outputs
        
        Args:
            model_outputs: List of raw text outputs from Qwen2.5-VL-7B model
            
        Returns:
            List of extracted re-edit instructions
        """
        instructions = []
        for output in model_outputs:
            instruction = self.extract_re_edit_instruction(output)
            instructions.append(instruction)
        return instructions
    
    def validate_instruction(self, instruction: str) -> bool:
        """
        Validate if the extracted instruction is reasonable
        
        Args:
            instruction: Extracted instruction to validate
            
        Returns:
            True if instruction is valid, False otherwise
        """
        if not instruction or len(instruction.strip()) < 5:
            return False
        
        # Check for common invalid patterns
        invalid_patterns = [
            r'^\s*$',  # Empty or whitespace only
            r'^[{}[\]]+$',  # Only brackets
            r'^["\']+$',  # Only quotes
            r'^[.,;:!?]+$',  # Only punctuation
        ]
        
        for pattern in invalid_patterns:
            if re.match(pattern, instruction):
                return False
        
        return True


def create_instruction_extractor(fallback_instruction: str = "Improve the image quality and consistency") -> InstructionExtractor:
    """
    Create an instruction extractor instance
    
    Args:
        fallback_instruction: Default instruction to use when extraction fails
        
    Returns:
        InstructionExtractor instance
    """
    return InstructionExtractor(fallback_instruction)


# Convenience functions for direct use
def extract_re_edit_instruction(model_output: str, fallback: str = "Improve the image quality and consistency") -> str:
    """
    Extract re-edit instruction from model output
    
    Args:
        model_output: Raw text output from Qwen2.5-VL-7B model
        fallback: Fallback instruction if extraction fails
        
    Returns:
        Extracted re-edit instruction
    """
    extractor = InstructionExtractor(fallback)
    return extractor.extract_re_edit_instruction(model_output)


def extract_batch_instructions(model_outputs: List[str], fallback: str = "Improve the image quality and consistency") -> List[str]:
    """
    Extract re-edit instructions from a batch of model outputs
    
    Args:
        model_outputs: List of raw text outputs from Qwen2.5-VL-7B model
        fallback: Fallback instruction if extraction fails
        
    Returns:
        List of extracted re-edit instructions
    """
    extractor = InstructionExtractor(fallback)
    return extractor.extract_batch_instructions(model_outputs)
