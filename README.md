# 🚀 Competitive Programming Python Template

A comprehensive Python template for competitive programming with built-in algorithms, data structures, and local testing support optimized for Spyder IDE.

## 📋 Table of Contents
- [Features](#features)
- [Quick Start](#quick-start)
- [Setup](#setup)
- [Usage](#usage)
- [Algorithms & Data Structures](#algorithms--data-structures)
- [Examples](#examples)
- [Tips & Tricks](#tips--tricks)
- [Contributing](#contributing)

## ✨ Features

- **🎯 Local Testing Support**: Easy file-based input/output for Spyder IDE
- **⚡ Fast I/O**: Optimized input/output for competitive programming
- **📚 Rich Algorithm Library**: 50+ pre-implemented algorithms and data structures
- **🔧 Ready-to-Use**: Just change the `solve()` function for each problem
- **📖 Comprehensive Documentation**: Detailed guide for every function

## 🏃 Quick Start

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/cp-python-template.git
cd cp-python-template
```

2. **Create `input.txt`** with your test input:
```
3
2 4
5 7
1 1
```

3. **Run the template**:
```bash
python main.py
```

4. **Before submitting** to online judge:
```python
LOCAL_TESTING = False  # Change from True to False
```

## 🛠️ Setup

### File Structure
```
cp-python-template/
├── main.py              # Main template file
├── input.txt            # Test input file
├── output.txt           # (Optional) Output file
├── test_setup.py        # Setup verification script
├── docs/
│   └── documentation.md # Detailed function documentation
└── examples/
    ├── graph_example.py
    ├── dp_example.py
    └── string_example.py
```

### Requirements
- Python 3.6+
- No external dependencies (uses only standard library)

### For Spyder IDE Users
1. Open `main.py` in Spyder
2. Ensure `input.txt` is in the same directory
3. Run directly - input will be read from file automatically

### For VS Code / Other IDEs
The template works with any Python IDE. Just ensure the working directory contains `input.txt`.

## 📝 Usage

### Basic Workflow

1. **Copy the problem's sample input** to `input.txt`
2. **Implement your solution** in the `solve()` function:
```python
def solve():
    n = read_int()
    arr = read_ints()
    # Your solution here
    print(result)
```

3. **Run and test** locally
4. **Submit** (remember to set `LOCAL_TESTING = False`)

### Reading Input

```python
# Single integer
n = read_int()

# Multiple integers
a, b, c = read_ints()

# Array of integers
arr = read_ints()

# String
s = read_str()

# Multiple strings
words = read_strs()
```

### Multiple Test Cases

```python
def main():
    t = read_int()  # Number of test cases
    for _ in range(t):
        solve()
```

## 🧮 Algorithms & Data Structures

### Sorting Algorithms
- **Built-in**: `sorted()`, `sort()`
- **Merge Sort**: Stable, O(n log n)
- **Quick Sort**: Average O(n log n)
- **Heap Sort**: O(n log n), in-place
- **Counting Sort**: O(n + k) for integers
- **Radix Sort**: O(d × (n + k))
- **Bucket Sort**: O(n + k) for uniform distribution

### Math Utilities
- GCD, LCM
- Prime checking, factorization
- Sieve of Eratosthenes
- Modular arithmetic (power, inverse, combinations)
- Fast exponentiation

### Graph Algorithms
- DFS, BFS
- Dijkstra's shortest path
- Bellman-Ford
- Floyd-Warshall
- Topological sort
- Union-Find (DSU)

### Data Structures
- **Segment Tree**: Range queries and updates
- **Fenwick Tree**: Prefix sums
- **Trie**: String operations
- **DSU**: Disjoint Set Union

### String Algorithms
- Z-algorithm
- KMP pattern matching
- Rolling hash

## 💡 Examples

### Example 1: Graph Problem
```python
def solve():
    n, m = read_ints()  # nodes, edges
    edges = []
    for _ in range(m):
        u, v, w = read_ints()
        edges.append((u, v, w))
    
    graph = build_weighted_graph(n, edges)
    distances = dijkstra(graph, 0)
    
    for d in distances:
        print(d if d != INF else -1)
```

### Example 2: Dynamic Programming
```python
def solve():
    n = read_int()
    arr = read_ints()
    
    # DP array
    dp = [0] * (n + 1)
    
    # Your DP logic here
    for i in range(n):
        dp[i + 1] = max(dp[i], dp[i] + arr[i])
    
    print(dp[n])
```

### Example 3: String Matching
```python
def solve():
    text = read_str()
    pattern = read_str()
    
    positions = kmp_pattern_search(text, pattern)
    
    print(len(positions))
    if positions:
        print(*positions)
```

## 🎯 Tips & Tricks

### Performance Tips
1. Use `sys.stdin.readline()` for faster input (already configured)
2. Use `collections.deque` for BFS instead of list
3. Use `heapq` for priority queue operations
4. Avoid deep recursion - use iteration when possible

### Debugging Tips
1. Use `print(..., file=sys.stderr)` for debug output
2. Test with edge cases (n=1, empty arrays, etc.)
3. Check for integer overflow with large numbers
4. Verify array bounds

### Common Pitfalls
- Remember to change `LOCAL_TESTING = False` before submitting
- Check if the problem uses 0-indexed or 1-indexed arrays
- Be careful with floating-point precision
- Watch out for TLE with recursive solutions

## 🤝 Contributing

Contributions are welcome! Feel free to:
- Add new algorithms
- Improve existing implementations
- Fix bugs
- Enhance documentation

### How to Contribute
1. Fork the repository
2. Create your feature branch (`git checkout -b feature/NewAlgorithm`)
3. Commit your changes (`git commit -m 'Add new algorithm'`)
4. Push to the branch (`git push origin feature/NewAlgorithm`)
5. Open a Pull Request

## 📜 License

This project is licensed under the MIT License - feel free to use it for your competitive programming journey!

## 🌟 Acknowledgments

- Inspired by competitive programming communities
- Algorithm implementations based on standard references
- Special thanks to all contributors

---

**Happy Coding! 🚀**

If you find this template helpful, please give it a ⭐ star!
