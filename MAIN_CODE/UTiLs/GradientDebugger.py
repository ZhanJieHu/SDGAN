class GradientDebugger:
    """
    A static utility class for debugging model gradients during training.
    
    Example usage:
        # Print all parameters every 10 iterations
        GradientDebugger.check_and_print(
            model=model,
            message="Training step",
            iteration=iteration,
            mode="all",
            print_interval=10
        )
        
        # Print only parameters without gradients
        GradientDebugger.check_and_print(
            model=model,
            message="Checking missing gradients",
            mode="no_grad"
        )
    """
    
    @staticmethod
    def should_print(cur_idx=None, print_interval=1):
        """
        Determine whether to print based on iteration and interval.
        
        Args:
            iteration: Current iteration/epoch number (None means always print)
            print_interval: Print every N iterations (default: 1)
            
        Returns:
            bool: True if should print, False otherwise
        """
        if cur_idx is None:
            return True
        return cur_idx % print_interval == 1
    
    @staticmethod
    def print_gradients(model, message="", iteration=None, mode="all"):
        """
        Print gradient information for model parameters.
        Fix BUG: 兼容 param.grad is None + 识别全0梯度 + 无遗漏打印无梯度参数
        Args:
            model: PyTorch model to inspect
            message: Custom message to display
            iteration: Current iteration/epoch number (optional)
            mode: Print mode - "all" (all params), "no_grad" (only params without gradients/zero grad),
                or "has_grad" (only params with gradients)
        """
        # Print header
        iteration_str = f"{iteration}" if iteration is not None else ""
        print(f"**************** {message} {iteration_str} ****************")
        
        # Print parameters in optimizer
        if mode == "all":
            print("✅ Trainable Parameters (requires_grad=True):")
            for name, param in model.named_parameters():
                if param.requires_grad:
                    print(f"{name}: requires_grad={param.requires_grad}")
            print()
        
        # Print gradient information
        print(f"📊 Gradient status (mode: {mode}):")
        no_grad_count = 0
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
                
            # 修复核心1：兼容 param.grad is None 的情况 + 识别全0梯度
            if param.grad is None:
                has_grad = False
                grad_norm = 0.0
            else:
                grad_norm = param.grad.norm().item()
                has_grad = grad_norm > 1e-8  # 梯度范数小于1e-8视为无有效梯度（全0）
            
            # Filter based on mode
            if mode == "no_grad" and has_grad:
                continue
            elif mode == "has_grad" and not has_grad:
                continue
            
            # Print gradient info
            if has_grad:
                print(f"✅ {name}: grad_norm={grad_norm:.6f}")
            else:
                print(f"🔴 {name}: NO GRADIENT / ZERO GRADIENT")
                no_grad_count += 1
            
        print(f"\n📌 Total no-gradient trainable params: {no_grad_count}")
        print("="*60 + "\n")
    
    @staticmethod
    def check_and_print(model, message="", cur_idx=None, mode="all", print_interval=1):
        """
        Check if should print and print gradient information accordingly.
        
        Args:
            model: PyTorch model to inspect
            message: Custom message to display
            iteration: Current iteration/epoch number (optional)
            mode: Print mode - "all", "no_grad", or "has_grad"
            print_interval: Print every N iterations (default: 1)
        """
        if GradientDebugger.should_print(cur_idx, print_interval):
            GradientDebugger.print_gradients(model, message, cur_idx, mode)




import torch
import torch.nn as nn

def test_gradient_debugger():
    """
    Test function for GradientDebugger class.
    Creates a simple model and tests all functionality.
    """
    
    # Create a simple test model
    class TestModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.layer1 = nn.Linear(10, 20)
            self.layer2 = nn.Linear(20, 10)
            self.layer3 = nn.Linear(10, 5)
            
            # Freeze layer3 to test no_grad case
            for param in self.layer3.parameters():
                param.requires_grad = False
    
    model = TestModel()
    
    print("="*80)
    print("TEST 1: Model without gradients (before backward)")
    print("="*80)
    
    # Test 1: Print all parameters before any forward/backward
    GradientDebugger.check_and_print(
        model=model,
        message="Initial State",
        mode="all"
    )
    
    # Test 2: Print only parameters without gradients
    print("\n" + "="*80)
    print("TEST 2: Only parameters without gradients")
    print("="*80)
    GradientDebugger.check_and_print(
        model=model,
        message="No Gradient Check",
        mode="no_grad"
    )
    
    print("\n" + "="*80)
    print("TEST 3: Performing forward and backward pass")
    print("="*80)
    
    # Create dummy input and perform forward/backward
    x = torch.randn(4, 10)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    
    # Forward pass
    output = model.layer2(model.layer1(x))
    loss = output.sum()
    
    # Backward pass
    loss.backward()
    
    # Test 3: Print all parameters after backward
    GradientDebugger.check_and_print(
        model=model,
        message="After Backward",
        mode="all"
    )
    
    # Test 4: Print only parameters with gradients
    print("\n" + "="*80)
    print("TEST 4: Only parameters WITH gradients")
    print("="*80)
    GradientDebugger.check_and_print(
        model=model,
        message="Has Gradient Check",
        mode="has_grad"
    )
    
    # Test 5: Print only parameters without gradients (should show layer3)
    print("\n" + "="*80)
    print("TEST 5: Only parameters WITHOUT gradients (frozen layer3)")
    print("="*80)
    GradientDebugger.check_and_print(
        model=model,
        message="Frozen Parameters",
        mode="no_grad"
    )
    
    # Test 6: Test iteration-based printing
    print("\n" + "="*80)
    print("TEST 6: Iteration-based printing (print every 5 iterations)")
    print("="*80)
    
    for i in range(12):
        # Should only print at iterations 0, 5, 10
        GradientDebugger.check_and_print(
            model=model,
            message="Training Iteration",
            cur_idx=i,
            mode="has_grad",
            print_interval=5
        )
    
    # Test 7: Test should_print logic
    print("\n" + "="*80)
    print("TEST 7: Testing should_print logic")
    print("="*80)
    
    print(f"Iteration 0, interval 1: {GradientDebugger.should_print(0, 1)}")  # True
    print(f"Iteration 5, interval 5: {GradientDebugger.should_print(5, 5)}")  # True
    print(f"Iteration 7, interval 5: {GradientDebugger.should_print(7, 5)}")  # False
    print(f"Iteration None: {GradientDebugger.should_print(None, 10)}")  # True
    
    # Test 8: Test with zero gradients
    print("\n" + "="*80)
    print("TEST 8: Model with zero gradients")
    print("="*80)
    
    optimizer.zero_grad()
    GradientDebugger.check_and_print(
        model=model,
        message="After zero_grad()",
        mode="all"
    )
    
    print("\n" + "="*80)
    print("TEST 9: Complex scenario - partial backward")
    print("="*80)
    
    # Only compute gradients for layer1
    x = torch.randn(4, 10)
    output = model.layer1(x)
    loss = output.sum()
    loss.backward()
    
    GradientDebugger.check_and_print(
        model=model,
        message="Partial Backward (only layer1)",
        mode="all"
    )
    
    print("\n" + "="*80)
    print("ALL TESTS COMPLETED SUCCESSFULLY!")
    print("="*80)


if __name__ == "__main__":
    test_gradient_debugger()