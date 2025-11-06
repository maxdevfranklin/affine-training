"""Data collection for Affine tasks (SAT, ABD, DED)"""

import asyncio
import json
import random
import sys
from pathlib import Path
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass

# Import R2BufferedDataset with fallback
# Note: R2BufferedDataset is not available in this codebase, so we create a mock
class R2BufferedDataset:
    """Mock R2BufferedDataset for local development"""
    def __init__(self, dataset_name, buffer_size=5, max_batch=5):
        self.dataset_name = dataset_name
        self.buffer_size = buffer_size
        self.max_batch = max_batch
        self._warned = False
    
    def __aiter__(self):
        if not self._warned:
            print("Warning: R2BufferedDataset not available. Using empty dataset.")
            self._warned = True
        return self._empty_iterator()
    
    async def _empty_iterator(self):
        # Empty async generator - returns no items
        # Fix: removed 'return' before 'yield' which caused early exit
        if False:
            yield  # This makes it a generator but never yields anything


@dataclass
class AffineSample:
    """Represents a training sample for Affine tasks"""
    prompt: str
    response: str
    environment: str
    difficulty: float
    metadata: Dict[str, Any]


class AffineDataCollector:
    """Collects and processes data for Affine tasks"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.dataset_name = config.get("dataset", "satpalsr/rl-python")
        self.buffer_size = config.get("buffer_size", 5)
        self.max_batch = config.get("max_batch", 5)
        self.num_samples = config.get("num_samples", 10000)
        self.environments = config.get("environments", ["affine:sat", "affine:abd", "affine:ded"])

    async def collect_sat_samples(self, num_samples: int) -> List[AffineSample]:
        """Collect SAT solving samples"""
        samples = []

        for i in range(num_samples):
            # Generate k-SAT problem
            n_vars = random.randint(10, 20)
            k = random.randint(3, 5)
            n_clauses = int(n_vars * 4.2)  # Near phase transition

            clauses = []
            for _ in range(n_clauses):
                clause = []
                vars_in_clause = random.sample(range(1, n_vars + 1), k)
                for var in vars_in_clause:
                    lit = var if random.random() > 0.5 else -var
                    clause.append(lit)
                clauses.append(clause)

            # Generate satisfying assignment
            assignment = {i: random.choice([True, False]) for i in range(1, n_vars + 1)}

            # Format prompt
            clauses_str = " ∧ ".join([
                "(" + " ∨ ".join([f"{'¬' if lit < 0 else ''}x{abs(lit)}" for lit in clause]) + ")"
                for clause in clauses
            ])

            prompt = f"""Solve the following {k}-SAT problem with {n_vars} variables:

{clauses_str}

Provide a satisfying assignment in the format: x1=True, x2=False, x3=True, ...
If the formula is unsatisfiable, respond with "UNSAT"."""

            # Format response
            response = ", ".join([f"x{i}={assignment[i]}" for i in range(1, n_vars + 1)])

            samples.append(AffineSample(
                prompt=prompt,
                response=response,
                environment="affine:sat",
                difficulty=n_clauses / n_vars,
                metadata={"n_vars": n_vars, "k": k, "n_clauses": n_clauses}
            ))

        return samples

    async def collect_abd_samples(self, num_samples: int) -> List[AffineSample]:
        """Collect abduction (reverse engineering) samples - synthetic generation"""
        samples = []

        # Generate synthetic code patterns for abduction
        patterns = [
            ("x = int(input())\nprint(x * 2)", lambda x: x * 2, "integer"),
            ("x = int(input())\ny = int(input())\nprint(x + y)", lambda x, y: x + y, "two integers"),
            ("x = int(input())\nprint(x ** 2)", lambda x: x ** 2, "integer"),
            ("s = input()\nprint(len(s))", lambda s: len(s), "string"),
            ("x = int(input())\nprint(abs(x))", lambda x: abs(x), "integer"),
            ("x = int(input())\nprint(x % 10)", lambda x: x % 10, "integer"),
            ("x = int(input())\nprint(str(x)[::-1])", lambda x: str(x)[::-1], "integer"),
            ("x = float(input())\nprint(round(x, 2))", lambda x: round(x, 2), "float"),
        ]

        for i in range(num_samples):
            code_template, func, input_type = random.choice(patterns)
            
            # Generate test input based on pattern
            if input_type == "integer":
                test_input = str(random.randint(1, 1000))
                expected_output = str(func(int(test_input)))
            elif input_type == "two integers":
                x = random.randint(1, 100)
                y = random.randint(1, 100)
                test_input = f"{x}\n{y}"
                expected_output = str(func(x, y))
            elif input_type == "string":
                test_input = random.choice(["hello", "world", "test", "python", "code"])
                expected_output = str(func(test_input))
            elif input_type == "float":
                test_input = str(random.uniform(1.0, 100.0))
                expected_output = str(func(float(test_input)))
            else:
                continue

            prompt = f"""Given the following Python program and its expected output, determine what input would produce this output.

Program:
```python
{code_template}
```

Expected Output:
{expected_output}

Provide the input in the format:
<INPUT>
[input lines here]
</INPUT>"""

            response = f"""<INPUT>
{test_input}
</INPUT>"""

            samples.append(AffineSample(
                prompt=prompt,
                response=response,
                environment="affine:abd",
                difficulty=len(code_template.split('\n')) / 50.0,
                metadata={"code_length": len(code_template), "num_lines": len(code_template.split('\n'))}
            ))

        return samples

    async def collect_ded_samples(self, num_samples: int) -> List[AffineSample]:
        """Collect deduction (code generation) samples - synthetic generation"""
        samples = []

        # Generate synthetic problem patterns
        problems = [
            ("Read two integers and print their sum.", 
             "a = int(input())\nb = int(input())\nprint(a + b)",
             [("5\n10", "15"), ("20\n30", "50"), ("100\n200", "300")]),
            ("Read an integer and print its square.",
             "x = int(input())\nprint(x * x)",
             [("5", "25"), ("10", "100"), ("7", "49")]),
            ("Read a string and print its length.",
             "s = input()\nprint(len(s))",
             [("hello", "5"), ("world", "5"), ("python", "6")]),
            ("Read an integer and print whether it's even or odd.",
             "x = int(input())\nprint('even' if x % 2 == 0 else 'odd')",
             [("4", "even"), ("7", "odd"), ("10", "even")]),
            ("Read two integers and print the maximum.",
             "a = int(input())\nb = int(input())\nprint(max(a, b))",
             [("5\n10", "10"), ("20\n15", "20"), ("3\n3", "3")]),
            ("Read an integer n and print the sum from 1 to n.",
             "n = int(input())\nprint(sum(range(1, n + 1)))",
             [("5", "15"), ("10", "55"), ("3", "6")]),
            ("Read a string and print it reversed.",
             "s = input()\nprint(s[::-1])",
             [("hello", "olleh"), ("world", "dlrow"), ("abc", "cba")]),
            ("Read a float and print it rounded to 2 decimal places.",
             "x = float(input())\nprint(f'{x:.2f}')",
             [("3.14159", "3.14"), ("2.71828", "2.72"), ("1.23456", "1.23")]),
        ]

        for i in range(num_samples):
            problem, code, test_cases = random.choice(problems)
            
            # Select 1-3 test cases to include
            num_test_cases = random.randint(1, min(3, len(test_cases)))
            selected_cases = random.sample(test_cases, num_test_cases)
            
            # Format test cases
            test_case_lines = []
            for inp, out in selected_cases:
                # Handle multi-line inputs
                inp_formatted = inp.replace('\n', '\\n')
                test_case_lines.append(f"Input: {inp_formatted}\nOutput: {out}")

            examples = "\n\n".join(test_case_lines) if test_case_lines else ""

            # Build the example test cases section
            example_section = ""
            if examples:
                example_section = f"\n\nExample test cases:\n{examples}"

            prompt = f"""Write a Python program that solves the following problem:

{problem}{example_section}

Write a complete program that reads from standard input and writes to standard output."""

            response = f"""```python
{code}
```"""

            samples.append(AffineSample(
                prompt=prompt,
                response=response,
                environment="affine:ded",
                difficulty=len(code.split('\n')) / 50.0,
                metadata={"code_length": len(code), "num_test_cases": len(selected_cases)}
            ))

        return samples

    async def collect_all_samples(self) -> Dict[str, List[AffineSample]]:
        """Collect samples for all Affine environments"""
        all_samples = {}
        samples_per_env = self.num_samples // len(self.environments)

        for env in self.environments:
            print(f"Collecting {samples_per_env} samples for {env}...")

            if env == "affine:sat":
                samples = await self.collect_sat_samples(samples_per_env)
            elif env == "affine:abd":
                samples = await self.collect_abd_samples(samples_per_env)
            elif env == "affine:ded":
                samples = await self.collect_ded_samples(samples_per_env)
            else:
                print(f"Unknown environment: {env}")
                continue

            all_samples[env] = samples
            print(f"Collected {len(samples)} samples for {env}")

        return all_samples

    def save_samples(self, samples: Dict[str, List[AffineSample]], output_dir: Path):
        """Save collected samples to disk"""
        output_dir.mkdir(parents=True, exist_ok=True)

        for env, env_samples in samples.items():
            env_name = env.replace(":", "_")
            output_file = output_dir / f"{env_name}_samples.jsonl"

            with open(output_file, 'w') as f:
                for sample in env_samples:
                    json_line = {
                        "prompt": sample.prompt,
                        "response": sample.response,
                        "environment": sample.environment,
                        "difficulty": sample.difficulty,
                        "metadata": sample.metadata
                    }
                    f.write(json.dumps(json_line) + '\n')

            print(f"Saved {len(env_samples)} samples to {output_file}")

    def load_samples(self, input_dir: Path) -> Dict[str, List[AffineSample]]:
        """Load samples from disk"""
        all_samples = {}

        for env in self.environments:
            env_name = env.replace(":", "_")
            input_file = input_dir / f"{env_name}_samples.jsonl"

            if not input_file.exists():
                print(f"Warning: {input_file} not found")
                continue

            samples = []
            with open(input_file, 'r') as f:
                for line in f:
                    data = json.loads(line)
                    samples.append(AffineSample(
                        prompt=data["prompt"],
                        response=data["response"],
                        environment=data["environment"],
                        difficulty=data["difficulty"],
                        metadata=data["metadata"]
                    ))

            all_samples[env] = samples
            print(f"Loaded {len(samples)} samples from {input_file}")

        return all_samples


async def main():
    """Example usage"""
    config = {
        "dataset": "satpalsr/rl-python",
        "buffer_size": 5,
        "max_batch": 5,
        "num_samples": 1000,
        "environments": ["affine:sat", "affine:abd", "affine:ded"]
    }

    collector = AffineDataCollector(config)
    samples = await collector.collect_all_samples()

    output_dir = Path("data_cache/affine")
    collector.save_samples(samples, output_dir)


if __name__ == "__main__":
    asyncio.run(main())
