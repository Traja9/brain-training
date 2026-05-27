"""
===========================================================
MOTOR CORTEX v1.0 - ACTION & EXECUTION LAYER
-----------------------------------------------------------
Brain's motor cortex for executing commands and actions.
Uses Cerebrum_Ultra for decision making.
===========================================================
"""

import os
import sys
import json
import subprocess
import time
from typing import Optional, List, Dict, Any, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from datetime import datetime
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# IMPORT CEREBRUM_ULTRA (Reasoning Module)
# ============================================================
try:
    from Cerebrum_Ultra import (
        NeuroGenesis,
        NeuroGenesisConfig,
        create_neurogenesis
    )
    CEREBRUM_AVAILABLE = True
except ImportError:
    print("⚠️  Warning: Cerebrum_Ultra not available")
    CEREBRUM_AVAILABLE = False


# ============================================================
# ACTION TYPES & CONFIGURATIONS
# ============================================================

class ActionType(Enum):
    """Types of executable actions"""
    PRINT = "print"
    FILE_READ = "file_read"
    FILE_WRITE = "file_write"
    FILE_DELETE = "file_delete"
    SYSTEM_COMMAND = "system_command"
    PYTHON_EXECUTE = "python_execute"
    API_CALL = "api_call"
    DEVICE_CONTROL = "device_control"
    NETWORK_REQUEST = "network_request"


class OutputFormat(Enum):
    """Output formatting options"""
    TEXT = "text"
    JSON = "json"
    FORMATTED = "formatted"
    TABLE = "table"


@dataclass
class ExecutionContext:
    """Execution context and metadata"""
    action_type: ActionType
    command: str = ""
    timestamp: float = field(default_factory=time.time)
    duration: float = 0.0
    success: bool = False
    status_code: int = 0
    output: Any = None
    error: Optional[str] = None
    logs: List[str] = field(default_factory=list)


# ============================================================
# EXECUTION ENGINES
# ============================================================

class CommandExecutor:
    """Execute system commands safely"""
    
    def __init__(self, timeout: int = 30):
        self.timeout = timeout
        self.execution_history: List[ExecutionContext] = []
    
    def execute(self, command: str, shell: bool = False) -> ExecutionContext:
        """Execute system command"""
        context = ExecutionContext(
            action_type=ActionType.SYSTEM_COMMAND,
            command=command
        )
        start_time = time.time()
        
        try:
            print(f"🔧 Executing: {command}")
            
            result = subprocess.run(
                command,
                shell=shell,
                capture_output=True,
                timeout=self.timeout,
                text=True
            )
            
            context.success = result.returncode == 0
            context.status_code = result.returncode
            context.output = result.stdout
            
            if result.returncode != 0:
                context.error = result.stderr
                print(f"⚠️  Command returned code {result.returncode}")
            else:
                print(f"✅ Success")
            
        except subprocess.TimeoutExpired:
            context.error = f"Timeout after {self.timeout}s"
            print(f"⏱️  {context.error}")
        
        except Exception as e:
            context.error = str(e)
            print(f"❌ Error: {context.error}")
        
        finally:
            context.duration = time.time() - start_time
            self.execution_history.append(context)
        
        return context


class FileOperator:
    """File I/O operations"""
    
    def __init__(self, base_path: str = "./data"):
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        self.operation_history: List[ExecutionContext] = []
    
    def read(self, filename: str) -> ExecutionContext:
        """Read file"""
        context = ExecutionContext(
            action_type=ActionType.FILE_READ,
            command=f"read {filename}"
        )
        start_time = time.time()
        
        try:
            filepath = self.base_path / filename
            
            if not filepath.exists():
                context.error = f"File not found: {filename}"
                print(f"❌ {context.error}")
                return context
            
            print(f"📖 Reading: {filename}")
            
            with open(filepath, 'r') as f:
                content = f.read()
            
            context.success = True
            context.status_code = 0
            context.output = content
            print(f"✅ Read {len(content)} bytes")
            
        except Exception as e:
            context.error = str(e)
            print(f"❌ Error: {context.error}")
        
        finally:
            context.duration = time.time() - start_time
            self.operation_history.append(context)
        
        return context
    
    def write(self, filename: str, content: str, 
              append: bool = False) -> ExecutionContext:
        """Write to file"""
        context = ExecutionContext(
            action_type=ActionType.FILE_WRITE,
            command=f"write {filename}"
        )
        start_time = time.time()
        
        try:
            filepath = self.base_path / filename
            filepath.parent.mkdir(parents=True, exist_ok=True)
            
            mode = 'a' if append else 'w'
            print(f"💾 Writing to: {filename}")
            
            with open(filepath, mode) as f:
                f.write(content)
            
            context.success = True
            context.status_code = 0
            context.output = filepath
            print(f"✅ Written {len(content)} bytes")
            
        except Exception as e:
            context.error = str(e)
            print(f"❌ Error: {context.error}")
        
        finally:
            context.duration = time.time() - start_time
            self.operation_history.append(context)
        
        return context
    
    def delete(self, filename: str) -> ExecutionContext:
        """Delete file"""
        context = ExecutionContext(
            action_type=ActionType.FILE_DELETE,
            command=f"delete {filename}"
        )
        start_time = time.time()
        
        try:
            filepath = self.base_path / filename
            
            if not filepath.exists():
                context.error = f"File not found: {filename}"
                return context
            
            print(f"🗑️  Deleting: {filename}")
            filepath.unlink()
            
            context.success = True
            context.status_code = 0
            print(f"✅ Deleted")
            
        except Exception as e:
            context.error = str(e)
            print(f"❌ Error: {context.error}")
        
        finally:
            context.duration = time.time() - start_time
            self.operation_history.append(context)
        
        return context


class DisplayOutput:
    """Display/Print operations"""
    
    def __init__(self):
        self.display_history: List[ExecutionContext] = []
    
    def print(self, content: Any, format: OutputFormat = OutputFormat.TEXT,
              title: Optional[str] = None) -> ExecutionContext:
        """Display output"""
        context = ExecutionContext(
            action_type=ActionType.PRINT,
            command=f"print {title or 'output'}"
        )
        
        try:
            print(f"📺 Displaying output...")
            
            if title:
                print(f"\n{'='*60}")
                print(f"{title}")
                print(f"{'='*60}")
            
            if format == OutputFormat.JSON and isinstance(content, dict):
                print(json.dumps(content, indent=2))
            
            elif format == OutputFormat.TABLE and isinstance(content, list):
                for row in content:
                    print(row)
            
            else:
                print(content)
            
            context.success = True
            context.status_code = 0
            context.output = content
            print(f"✅ Displayed")
            
        except Exception as e:
            context.error = str(e)
            print(f"❌ Error: {context.error}")
        
        finally:
            self.display_history.append(context)
        
        return context


# ============================================================
# MAIN MOTOR CORTEX CLASS
# ============================================================

class MotorCortex:
    """
    Motor Cortex - Action Execution Layer
    
    Takes decisions from Cerebrum_Ultra and executes them.
    Bridges reasoning (Cerebrum) with action (Execution).
    """
    
    def __init__(self, cerebrum: Optional[NeuroGenesis] = None):
        """
        Initialize Motor Cortex
        
        Args:
            cerebrum: NeuroGenesis instance for reasoning
        """
        self.cerebrum = cerebrum
        self.command_executor = CommandExecutor()
        self.file_operator = FileOperator()
        self.display = DisplayOutput()
        
        self.action_history: List[ExecutionContext] = []
        self.execution_count = 0
        
        print(f"\n🧠 Motor Cortex Initialized")
        print(f"   ├─ Command Executor: Ready")
        print(f"   ├─ File Operator: Ready")
        print(f"   └─ Display Output: Ready\n")
    
    # --------------------------------------------------------
    # DECISION-TO-ACTION PIPELINE
    # --------------------------------------------------------
    
    def execute_from_reasoning(self, reasoning: str, 
                               context_data: Optional[Dict] = None) -> Dict:
        """
        Execute action based on Cerebrum reasoning
        
        Args:
            reasoning: Reasoning/decision from Cerebrum
            context_data: Additional context
            
        Returns:
            Execution result
        """
        print(f"\n🎯 Executing from reasoning: {reasoning}")
        
        # Parse reasoning to action
        action = self._parse_reasoning(reasoning)
        
        # Execute action
        result = self._execute_action(action, context_data)
        
        # Record action
        self.action_history.append(result)
        self.execution_count += 1
        
        return {
            'reasoning': reasoning,
            'action': action,
            'execution': result,
            'success': result.success
        }
    
    def _parse_reasoning(self, reasoning: str) -> Dict:
        """Parse reasoning string into actionable command"""
        reasoning = reasoning.lower()
        
        # Simple action parsing
        if 'print' in reasoning or 'display' in reasoning:
            return {'type': 'print', 'content': reasoning}
        
        elif 'file' in reasoning and 'read' in reasoning:
            filename = self._extract_filename(reasoning)
            return {'type': 'file_read', 'filename': filename}
        
        elif 'file' in reasoning and 'write' in reasoning:
            filename = self._extract_filename(reasoning)
            return {'type': 'file_write', 'filename': filename}
        
        elif 'command' in reasoning or 'execute' in reasoning:
            command = self._extract_command(reasoning)
            return {'type': 'system_command', 'command': command}
        
        else:
            return {'type': 'print', 'content': reasoning}
    
    def _extract_filename(self, text: str) -> str:
        """Extract filename from text"""
        import re
        match = re.search(r"['\"]([^'\"]+\.(txt|json|csv|py))['\"]", text)
        return match.group(1) if match else "output.txt"
    
    def _extract_command(self, text: str) -> str:
        """Extract command from text"""
        import re
        match = re.search(r"execute[:\s]+([^\n]+)", text)
        return match.group(1) if match else "echo 'command'"
    
    def _execute_action(self, action: Dict, 
                       context_data: Optional[Dict] = None) -> ExecutionContext:
        """Execute parsed action"""
        action_type = action.get('type', 'print')
        
        if action_type == 'print':
            return self.display.print(
                action.get('content', 'Action completed'),
                title="Motor Cortex Output"
            )
        
        elif action_type == 'file_read':
            return self.file_operator.read(action.get('filename', 'data.txt'))
        
        elif action_type == 'file_write':
            return self.file_operator.write(
                action.get('filename', 'output.txt'),
                str(action.get('content', ''))
            )
        
        elif action_type == 'system_command':
            return self.command_executor.execute(action.get('command', 'echo ok'))
        
        else:
            return ExecutionContext(action_type=ActionType.PRINT)
    
    # --------------------------------------------------------
    # INTEGRATED BRAIN PIPELINE
    # --------------------------------------------------------
    
    def process_with_cerebrum(self, input_data: np.ndarray) -> Dict:
        """
        Complete pipeline: Input → Cerebrum (reason) → Motor Cortex (act)
        
        Args:
            input_data: Input data for reasoning
            
        Returns:
            Complete processing result
        """
        if not self.cerebrum:
            print("❌ Cerebrum not connected!")
            return {'error': 'Cerebrum not available'}
        
        print(f"\n🧠 INTEGRATED BRAIN PIPELINE")
        print(f"{'='*60}")
        
        # Step 1: Input processing
        print(f"\n1️⃣  Input Received: shape {input_data.shape}")
        
        # Step 2: Reasoning (Cerebrum)
        print(f"\n2️⃣  Reasoning (Cerebrum_Ultra)...")
        predictions = self.cerebrum.predict(input_data)
        print(f"   ✅ Predictions: {predictions.shape}")
        print(f"   Values: {predictions.flatten()[:3]}...")
        
        # Step 3: Decision making
        print(f"\n3️⃣  Decision Making...")
        decision = self._make_decision(predictions, input_data)
        print(f"   ✅ Decision: {decision}")
        
        # Step 4: Execution (Motor Cortex)
        print(f"\n4️⃣  Execution (Motor Cortex)...")
        execution_result = self.execute_from_reasoning(decision)
        
        print(f"\n{'='*60}")
        print(f"✅ Pipeline Complete!")
        
        return {
            'input': input_data,
            'predictions': predictions,
            'decision': decision,
            'execution': execution_result
        }
    
    def _make_decision(self, predictions: np.ndarray, 
                      input_data: np.ndarray) -> str:
        """Make decision based on predictions"""
        mean_pred = np.mean(predictions)
        std_pred = np.std(predictions)
        max_pred = np.max(predictions)
        
        if max_pred > 0.8:
            return f"High confidence prediction detected. Display results."
        elif std_pred > 0.5:
            return f"High uncertainty. Store for review."
        else:
            return f"Normal prediction. Continue processing."
    
    # --------------------------------------------------------
    # BATCH OPERATIONS
    # --------------------------------------------------------
    
    def batch_execute(self, actions: List[Dict]) -> List[ExecutionContext]:
        """Execute multiple actions in sequence"""
        print(f"\n🔄 Batch Executing {len(actions)} actions...")
        results = []
        
        for i, action in enumerate(actions, 1):
            print(f"\n   [{i}/{len(actions)}] {action.get('type', 'unknown')}")
            result = self._execute_action(action)
            results.append(result)
        
        print(f"\n✅ Batch execution complete ({len(results)} actions)")
        return results
    
    # --------------------------------------------------------
    # STATISTICS & REPORTING
    # --------------------------------------------------------
    
    def get_statistics(self) -> Dict:
        """Get execution statistics"""
        total_time = sum(ctx.duration for ctx in self.action_history)
        successful = sum(1 for ctx in self.action_history if ctx.success)
        
        return {
            'total_actions': self.execution_count,
            'successful': successful,
            'failed': self.execution_count - successful,
            'success_rate': successful / max(1, self.execution_count),
            'total_time': total_time,
            'avg_time': total_time / max(1, self.execution_count)
        }
    
    def summary(self):
        """Display Motor Cortex summary"""
        print(f"\n{'='*60}")
        print(f"🧠 MOTOR CORTEX SUMMARY")
        print(f"{'='*60}")
        print(f"Cerebrum Connected: {'✅ Yes' if self.cerebrum else '❌ No'}")
        print(f"Actions Executed: {self.execution_count}")
        print(f"Command History: {len(self.command_executor.execution_history)}")
        print(f"File Operations: {len(self.file_operator.operation_history)}")
        print(f"Display Operations: {len(self.display.display_history)}")
        
        stats = self.get_statistics()
        print(f"\n📊 Statistics:")
        print(f"   Success Rate: {stats['success_rate']:.1%}")
        print(f"   Total Time: {stats['total_time']:.2f}s")
        print(f"   Avg Time/Action: {stats['avg_time']:.3f}s")
        print(f"{'='*60}\n")


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def create_motor_cortex(cerebrum: Optional[NeuroGenesis] = None) -> MotorCortex:
    """Factory function to create Motor Cortex"""
    return MotorCortex(cerebrum=cerebrum)


# ============================================================
# DEMONSTRATION
# ============================================================

if __name__ == "__main__":
    print("\n" + "="*60)
    print("🧠 MOTOR CORTEX v1.0 - DEMONSTRATION")
    print("="*60)
    
    # Create Motor Cortex
    motor_cortex = create_motor_cortex()
    
    # Demo 1: File operations
    print("\n" + "="*60)
    print("DEMO 1: FILE OPERATIONS")
    print("="*60)
    
    motor_cortex.file_operator.write(
        "test.txt",
        "Hello from Motor Cortex!\nThis is a test file."
    )
    
    motor_cortex.file_operator.read("test.txt")
    
    # Demo 2: Display output
    print("\n" + "="*60)
    print("DEMO 2: DISPLAY OUTPUT")
    print("="*60)
    
    data = {
        'action': 'test',
        'status': 'success',
        'timestamp': datetime.now().isoformat()
    }
    
    motor_cortex.display.print(data, OutputFormat.JSON, "Execution Result")
    
    # Demo 3: Command execution
    print("\n" + "="*60)
    print("DEMO 3: COMMAND EXECUTION")
    print("="*60)
    
    motor_cortex.command_executor.execute("echo 'Motor Cortex is alive!'", shell=True)
    
    # Demo 4: Summary
    motor_cortex.summary()
    
    print("\n✅ Motor Cortex Demonstration Complete!\n")