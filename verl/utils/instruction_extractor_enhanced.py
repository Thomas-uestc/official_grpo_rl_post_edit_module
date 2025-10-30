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
Enhanced Instruction Extractor with multi-tag support
"""

import json
import re
from typing import Dict, List, Optional, Any
import logging

logger = logging.getLogger(__name__)


class InstructionExtractorEnhanced:
    """
    Enhanced instruction extractor that supports extracting multiple Re_edit tags
    and concatenating them into a single instruction
    """
    
    def __init__(
        self, 
        fallback_instruction: str = "Improve the image quality and consistency",
        enable_multi_tag: bool = True,
        concatenation_separator: str = "; "
    ):
        """
        Initialize the enhanced instruction extractor
        
        Args:
            fallback_instruction: Default instruction to use when extraction fails
            enable_multi_tag: Whether to extract and concatenate multiple Re_edit tags
            concatenation_separator: Separator used to join multiple instructions
        """
        self.fallback_instruction = fallback_instruction
        self.enable_multi_tag = enable_multi_tag
        self.concatenation_separator = concatenation_separator
        
        # Precompile regex patterns
        self.xml_pattern = re.compile(r'<Re_edit[^>]*>(.*?)</Re_edit>', re.IGNORECASE | re.DOTALL)
        self.json_pattern = re.compile(r'"Re_Edit"\s*:\s*"([^"]+)"', re.IGNORECASE)
    
    def extract_re_edit_instruction(self, model_output: str) -> str:
        """
        Extract the Re_Edit instruction(s) from model output
        
        Args:
            model_output: Raw text output from Qwen2.5-VL-7B model
            
        Returns:
            Extracted re-edit instruction(s) or fallback instruction
        """
        try:
            # Method 1: Try to extract using XML tags (supports multi-tag)
            xml_instruction = self._extract_with_xml_tags(model_output)
            if xml_instruction:
                return xml_instruction
            
            # Method 2: Try to extract using JSON format
            json_instruction = self._extract_from_json(model_output)
            if json_instruction:
                return json_instruction
            
            # Method 3: Try other patterns
            regex_instruction = self._extract_with_regex(model_output)
            if regex_instruction:
                return regex_instruction
            
            logger.warning(f"Failed to extract instruction from: {model_output[:200]}...")
            return self.fallback_instruction
            
        except Exception as e:
            logger.error(f"Error extracting instruction: {e}")
            return self.fallback_instruction
    
    def _extract_with_xml_tags(self, text: str) -> Optional[str]:
        """
        Extract Re_Edit from XML tags, supporting multiple tags
        
        Args:
            text: Input text containing Re_edit tags
            
        Returns:
            Extracted instruction(s), concatenated if multiple tags found
        """
        try:
            # Find all matches
            matches = self.xml_pattern.findall(text)
            
            if not matches:
                return None
            
            if self.enable_multi_tag and len(matches) > 1:
                # Multiple tags found - extract and concatenate
                instructions = []
                for match in matches:
                    instruction = match.strip()
                    # Clean up newlines and extra spaces
                    instruction = re.sub(r'\s+', ' ', instruction)
                    instruction = instruction.strip()
                    
                    if instruction:  # Only add non-empty instructions
                        instructions.append(instruction)
                
                if instructions:
                    # Concatenate with separator
                    concatenated = self.concatenation_separator.join(instructions)
                    logger.info(f"Extracted {len(instructions)} Re_edit tags, concatenated into: {concatenated[:100]}...")
                    return concatenated
                else:
                    return None
            else:
                # Single tag or multi-tag disabled - return first match
                instruction = matches[0].strip()
                instruction = re.sub(r'\s+', ' ', instruction)
                return instruction.strip()
                
        except Exception as e:
            logger.debug(f"XML tag extraction failed: {e}")
            return None
    
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
                def handle_duplicates(pairs):
                    result = {}
                    for key, value in pairs:
                        if key not in result:
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
        """Extract Re_Edit using regex patterns (fallback)"""
        try:
            # Pattern: "Re_Edit": "instruction" (JSON format)
            match = self.json_pattern.search(text)
            if match:
                return match.group(1).strip()
            
            # Pattern: Re_Edit: instruction (without quotes)
            colon_pattern = r'Re_Edit\s*:\s*([^\n\r,}<]+?)(?=\n|$|,|\}|<)'
            match = re.search(colon_pattern, text, re.IGNORECASE)
            if match:
                instruction = match.group(1).strip()
                instruction = re.sub(r'^["\']|["\']$', '', instruction)
                return instruction.strip()
                
        except Exception as e:
            logger.debug(f"Regex extraction failed: {e}")
        
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
    
    def extract_all_tags(self, model_output: str) -> List[str]:
        """
        Extract all Re_edit tags as a list (without concatenation)
        
        Args:
            model_output: Raw text output from model
            
        Returns:
            List of all extracted Re_edit instructions
        """
        try:
            matches = self.xml_pattern.findall(model_output)
            
            instructions = []
            for match in matches:
                instruction = match.strip()
                instruction = re.sub(r'\s+', ' ', instruction)
                instruction = instruction.strip()
                
                if instruction:
                    instructions.append(instruction)
            
            return instructions if instructions else [self.fallback_instruction]
            
        except Exception as e:
            logger.error(f"Error extracting all tags: {e}")
            return [self.fallback_instruction]
    
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


# Convenience functions for backward compatibility
def create_instruction_extractor(
    fallback_instruction: str = "Improve the image quality and consistency",
    enable_multi_tag: bool = True,
    concatenation_separator: str = "; "
) -> InstructionExtractorEnhanced:
    """
    Create an enhanced instruction extractor instance
    
    Args:
        fallback_instruction: Default instruction to use when extraction fails
        enable_multi_tag: Whether to extract and concatenate multiple Re_edit tags
        concatenation_separator: Separator used to join multiple instructions
        
    Returns:
        InstructionExtractorEnhanced instance
    """
    return InstructionExtractorEnhanced(
        fallback_instruction=fallback_instruction,
        enable_multi_tag=enable_multi_tag,
        concatenation_separator=concatenation_separator
    )


def extract_re_edit_instruction(
    model_output: str, 
    fallback: str = "Improve the image quality and consistency",
    enable_multi_tag: bool = True
) -> str:
    """
    Extract re-edit instruction from model output
    
    Args:
        model_output: Raw text output from Qwen2.5-VL-7B model
        fallback: Fallback instruction if extraction fails
        enable_multi_tag: Whether to extract and concatenate multiple Re_edit tags
        
    Returns:
        Extracted re-edit instruction
    """
    extractor = InstructionExtractorEnhanced(fallback, enable_multi_tag=enable_multi_tag)
    return extractor.extract_re_edit_instruction(model_output)


def extract_batch_instructions(
    model_outputs: List[str], 
    fallback: str = "Improve the image quality and consistency",
    enable_multi_tag: bool = True
) -> List[str]:
    """
    Extract re-edit instructions from a batch of model outputs
    
    Args:
        model_outputs: List of raw text outputs from Qwen2.5-VL-7B model
        fallback: Fallback instruction if extraction fails
        enable_multi_tag: Whether to extract and concatenate multiple Re_edit tags
        
    Returns:
        List of extracted re-edit instructions
    """
    extractor = InstructionExtractorEnhanced(fallback, enable_multi_tag=enable_multi_tag)
    return extractor.extract_batch_instructions(model_outputs)

