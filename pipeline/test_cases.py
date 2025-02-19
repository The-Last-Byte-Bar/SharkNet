from typing import Dict, List, Optional, Union
import json
import os
from dataclasses import dataclass
from datetime import datetime

@dataclass
class TestCase:
    """Container for a single test case."""
    id: str
    input_text: str
    expected_output: Optional[str] = None
    category: str = "general"
    difficulty: str = "medium"
    tags: List[str] = None
    metadata: Dict = None

    def __post_init__(self):
        if self.tags is None:
            self.tags = []
        if self.metadata is None:
            self.metadata = {}

class TestCaseManager:
    """Manages test cases for both predefined and live evaluation."""
    
    def __init__(self, test_cases_dir: str = "test_cases"):
        self.test_cases_dir = test_cases_dir
        self.predefined_cases: Dict[str, TestCase] = {}
        self.live_cases: Dict[str, TestCase] = {}
        os.makedirs(test_cases_dir, exist_ok=True)
        self._load_predefined_cases()

    def _load_predefined_cases(self):
        """Load predefined test cases from files."""
        categories = [
            "basic_syntax",
            "smart_contracts",
            "error_handling",
            "complex_logic",
            "edge_cases"
        ]
        
        for category in categories:
            category_file = os.path.join(self.test_cases_dir, f"{category}.json")
            if os.path.exists(category_file):
                with open(category_file, 'r') as f:
                    cases = json.load(f)
                    for case in cases:
                        test_case = TestCase(
                            id=case['id'],
                            input_text=case['input_text'],
                            expected_output=case.get('expected_output'),
                            category=category,
                            difficulty=case.get('difficulty', 'medium'),
                            tags=case.get('tags', []),
                            metadata=case.get('metadata', {})
                        )
                        self.predefined_cases[test_case.id] = test_case

    def add_live_case(self, input_text: str, expected_output: Optional[str] = None,
                     category: str = "live", tags: List[str] = None) -> str:
        """Add a new live test case."""
        case_id = f"live_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        test_case = TestCase(
            id=case_id,
            input_text=input_text,
            expected_output=expected_output,
            category=category,
            tags=tags or []
        )
        self.live_cases[case_id] = test_case
        return case_id

    def get_test_cases(self, category: Optional[str] = None, 
                      difficulty: Optional[str] = None,
                      tags: Optional[List[str]] = None,
                      case_type: str = "all") -> List[TestCase]:
        """Get filtered test cases."""
        if case_type == "predefined":
            cases = self.predefined_cases.values()
        elif case_type == "live":
            cases = self.live_cases.values()
        else:
            cases = list(self.predefined_cases.values()) + list(self.live_cases.values())

        filtered_cases = []
        for case in cases:
            if category and case.category != category:
                continue
            if difficulty and case.difficulty != difficulty:
                continue
            if tags and not all(tag in case.tags for tag in tags):
                continue
            filtered_cases.append(case)

        return filtered_cases

    def save_test_case(self, test_case: TestCase, is_predefined: bool = False):
        """Save a test case to appropriate storage."""
        if is_predefined:
            category_file = os.path.join(self.test_cases_dir, f"{test_case.category}.json")
            cases = []
            if os.path.exists(category_file):
                with open(category_file, 'r') as f:
                    cases = json.load(f)

            # Update or append case
            case_dict = {
                'id': test_case.id,
                'input_text': test_case.input_text,
                'expected_output': test_case.expected_output,
                'difficulty': test_case.difficulty,
                'tags': test_case.tags,
                'metadata': test_case.metadata
            }
            
            for i, case in enumerate(cases):
                if case['id'] == test_case.id:
                    cases[i] = case_dict
                    break
            else:
                cases.append(case_dict)

            with open(category_file, 'w') as f:
                json.dump(cases, f, indent=2)
        else:
            self.live_cases[test_case.id] = test_case

    def create_test_suite(self, name: str, test_cases: List[TestCase]):
        """Create a named test suite from a list of test cases."""
        suite_file = os.path.join(self.test_cases_dir, f"suite_{name}.json")
        suite = {
            'name': name,
            'created_at': datetime.now().isoformat(),
            'test_cases': [
                {
                    'id': case.id,
                    'input_text': case.input_text,
                    'expected_output': case.expected_output,
                    'category': case.category,
                    'difficulty': case.difficulty,
                    'tags': case.tags,
                    'metadata': case.metadata
                }
                for case in test_cases
            ]
        }
        
        with open(suite_file, 'w') as f:
            json.dump(suite, f, indent=2)

    def load_test_suite(self, name: str) -> List[TestCase]:
        """Load a test suite by name."""
        suite_file = os.path.join(self.test_cases_dir, f"suite_{name}.json")
        if not os.path.exists(suite_file):
            raise FileNotFoundError(f"Test suite '{name}' not found")

        with open(suite_file, 'r') as f:
            suite = json.load(f)
            return [
                TestCase(
                    id=case['id'],
                    input_text=case['input_text'],
                    expected_output=case.get('expected_output'),
                    category=case.get('category', 'general'),
                    difficulty=case.get('difficulty', 'medium'),
                    tags=case.get('tags', []),
                    metadata=case.get('metadata', {})
                )
                for case in suite['test_cases']
            ] 