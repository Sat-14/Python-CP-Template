#!/usr/bin/env python3
import sys
import math
import bisect
import heapq
from collections import defaultdict, deque, Counter
from itertools import accumulate, permutations, combinations
from functools import lru_cache, reduce
from typing import List, Tuple, Dict, Set

# ============= INPUT CONFIGURATION =============
# Method 1: For Local Testing with File Input (Recommended for Spyder)
LOCAL_TESTING = True  # Set to False when submitting to online judge

if LOCAL_TESTING:
    # Read from input.txt file
    sys.stdin = open('input.txt', 'r')
    # Optional: Write output to file
    # sys.stdout = open('output.txt', 'w')
else:
    # Fast I/O for competitive programming judges
    input = lambda: sys.stdin.readline().rstrip()
    print = lambda *args, **kwargs: sys.stdout.write(' '.join(map(str, args)) + kwargs.get('end', '\n'))

# Method 2: Alternative - Direct file reading (uncomment if preferred)
"""
def read_from_file(filename='input.txt'):
    with open(filename, 'r') as f:
        return f.read().strip().split('\n')

# Usage:
if LOCAL_TESTING:
    input_lines = read_from_file()
    input_index = 0
    
    def input():
        global input_index
        if input_index < len(input_lines):
            line = input_lines[input_index]
            input_index += 1
            return line
        return ""
"""

# Constants
MOD = 10**9 + 7
INF = float('inf')
NINF = float('-inf')
YES = "YES"
NO = "NO"

# Common functions
def read_int() -> int:
    return int(input())

def read_ints() -> List[int]:
    return list(map(int, input().split()))

def read_str() -> str:
    return input()

def read_strs() -> List[str]:
    return input().split()

# Sorting Algorithms
def merge_sort(arr: List[int]) -> List[int]:
    """Stable O(n log n) sorting algorithm"""
    if len(arr) <= 1:
        return arr
    
    mid = len(arr) // 2
    left = merge_sort(arr[:mid])
    right = merge_sort(arr[mid:])
    
    result = []
    i = j = 0
    
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1
    
    result.extend(left[i:])
    result.extend(right[j:])
    return result

def quick_sort(arr: List[int]) -> List[int]:
    """Average O(n log n), worst O(n²) sorting algorithm"""
    if len(arr) <= 1:
        return arr
    
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    
    return quick_sort(left) + middle + quick_sort(right)

def heap_sort(arr: List[int]) -> List[int]:
    """O(n log n) sorting using heap data structure"""
    result = []
    heap = arr.copy()
    heapq.heapify(heap)
    
    while heap:
        result.append(heapq.heappop(heap))
    
    return result

def counting_sort(arr: List[int], max_val: int = None) -> List[int]:
    """O(n + k) sorting for non-negative integers, k = range of values"""
    if not arr:
        return arr
    
    if max_val is None:
        max_val = max(arr)
    
    count = [0] * (max_val + 1)
    for num in arr:
        count[num] += 1
    
    result = []
    for i in range(len(count)):
        result.extend([i] * count[i])
    
    return result

def radix_sort(arr: List[int]) -> List[int]:
    """O(d * (n + k)) sorting for non-negative integers, d = digits"""
    if not arr:
        return arr
    
    max_num = max(arr)
    exp = 1
    
    while max_num // exp > 0:
        arr = counting_sort_by_digit(arr, exp)
        exp *= 10
    
    return arr

def counting_sort_by_digit(arr: List[int], exp: int) -> List[int]:
    """Helper for radix sort"""
    n = len(arr)
    output = [0] * n
    count = [0] * 10
    
    for i in range(n):
        index = arr[i] // exp
        count[index % 10] += 1
    
    for i in range(1, 10):
        count[i] += count[i - 1]
    
    i = n - 1
    while i >= 0:
        index = arr[i] // exp
        output[count[index % 10] - 1] = arr[i]
        count[index % 10] -= 1
        i -= 1
    
    return output

def bucket_sort(arr: List[float], num_buckets: int = 10) -> List[float]:
    """O(n + k) average case sorting for uniformly distributed floats in [0, 1)"""
    if not arr:
        return arr
    
    buckets = [[] for _ in range(num_buckets)]
    
    for num in arr:
        bucket_index = int(num * num_buckets)
        if bucket_index == num_buckets:
            bucket_index -= 1
        buckets[bucket_index].append(num)
    
    for i in range(num_buckets):
        buckets[i].sort()
    
    result = []
    for bucket in buckets:
        result.extend(bucket)
    
    return result

def custom_sort(arr: List, key=None, reverse: bool = False) -> List:
    """Sort with custom key function"""
    return sorted(arr, key=key, reverse=reverse)

# Math utilities
def gcd(a: int, b: int) -> int:
    while b:
        a, b = b, a % b
    return a

def lcm(a: int, b: int) -> int:
    return a * b // gcd(a, b)

def prime_factors(n: int) -> List[int]:
    factors = []
    d = 2
    while d * d <= n:
        while n % d == 0:
            factors.append(d)
            n //= d
        d += 1
    if n > 1:
        factors.append(n)
    return factors

def is_prime(n: int) -> bool:
    if n < 2:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False
    for i in range(3, int(n**0.5) + 1, 2):
        if n % i == 0:
            return False
    return True

def sieve_of_eratosthenes(n: int) -> List[bool]:
    """Generate prime sieve up to n"""
    is_prime = [True] * (n + 1)
    is_prime[0] = is_prime[1] = False
    
    for i in range(2, int(n**0.5) + 1):
        if is_prime[i]:
            for j in range(i * i, n + 1, i):
                is_prime[j] = False
    
    return is_prime

def get_primes(n: int) -> List[int]:
    """Get all primes up to n"""
    sieve = sieve_of_eratosthenes(n)
    return [i for i in range(2, n + 1) if sieve[i]]

# Modular arithmetic
def mod_pow(base: int, exp: int, mod: int = MOD) -> int:
    result = 1
    base %= mod
    while exp > 0:
        if exp & 1:
            result = (result * base) % mod
        base = (base * base) % mod
        exp >>= 1
    return result

def mod_inverse(a: int, mod: int = MOD) -> int:
    return mod_pow(a, mod - 2, mod)

def mod_factorial(n: int, mod: int = MOD) -> int:
    """Calculate n! % mod"""
    result = 1
    for i in range(1, n + 1):
        result = (result * i) % mod
    return result

def mod_combination(n: int, r: int, mod: int = MOD) -> int:
    """Calculate nCr % mod"""
    if r > n or r < 0:
        return 0
    if r == 0 or r == n:
        return 1
    
    num = mod_factorial(n, mod)
    den = (mod_factorial(r, mod) * mod_factorial(n - r, mod)) % mod
    return (num * mod_inverse(den, mod)) % mod

# Binary search utilities
def binary_search_left(arr: List[int], target: int) -> int:
    """Find leftmost position where target can be inserted"""
    return bisect.bisect_left(arr, target)

def binary_search_right(arr: List[int], target: int) -> int:
    """Find rightmost position where target can be inserted"""
    return bisect.bisect_right(arr, target)

def binary_search_custom(low: int, high: int, predicate) -> int:
    """Custom binary search with predicate function"""
    while low < high:
        mid = (low + high) // 2
        if predicate(mid):
            high = mid
        else:
            low = mid + 1
    return low

# Graph utilities
def build_graph(n: int, edges: List[Tuple[int, int]], directed: bool = False) -> List[List[int]]:
    graph = [[] for _ in range(n)]
    for u, v in edges:
        graph[u].append(v)
        if not directed:
            graph[v].append(u)
    return graph

def build_weighted_graph(n: int, edges: List[Tuple[int, int, int]], directed: bool = False) -> List[List[Tuple[int, int]]]:
    graph = [[] for _ in range(n)]
    for u, v, w in edges:
        graph[u].append((v, w))
        if not directed:
            graph[v].append((u, w))
    return graph

def dfs(graph: List[List[int]], start: int, visited: Set[int] = None) -> List[int]:
    if visited is None:
        visited = set()
    
    result = []
    stack = [start]
    
    while stack:
        node = stack.pop()
        if node not in visited:
            visited.add(node)
            result.append(node)
            for neighbor in reversed(graph[node]):
                if neighbor not in visited:
                    stack.append(neighbor)
    
    return result

def bfs(graph: List[List[int]], start: int) -> List[int]:
    visited = set([start])
    queue = deque([start])
    result = []
    
    while queue:
        node = queue.popleft()
        result.append(node)
        
        for neighbor in graph[node]:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)
    
    return result

def dijkstra(graph: List[List[Tuple[int, int]]], start: int) -> List[int]:
    n = len(graph)
    dist = [INF] * n
    dist[start] = 0
    pq = [(0, start)]
    
    while pq:
        d, u = heapq.heappop(pq)
        if d > dist[u]:
            continue
        for v, w in graph[u]:
            if dist[u] + w < dist[v]:
                dist[v] = dist[u] + w
                heapq.heappush(pq, (dist[v], v))
    
    return dist

def bellman_ford(n: int, edges: List[Tuple[int, int, int]], start: int) -> Tuple[List[int], bool]:
    """Returns (distances, has_negative_cycle)"""
    dist = [INF] * n
    dist[start] = 0
    
    for _ in range(n - 1):
        for u, v, w in edges:
            if dist[u] != INF and dist[u] + w < dist[v]:
                dist[v] = dist[u] + w
    
    # Check for negative cycle
    for u, v, w in edges:
        if dist[u] != INF and dist[u] + w < dist[v]:
            return dist, True
    
    return dist, False

def floyd_warshall(n: int, edges: List[Tuple[int, int, int]]) -> List[List[int]]:
    """All pairs shortest paths"""
    dist = [[INF] * n for _ in range(n)]
    
    for i in range(n):
        dist[i][i] = 0
    
    for u, v, w in edges:
        dist[u][v] = min(dist[u][v], w)
    
    for k in range(n):
        for i in range(n):
            for j in range(n):
                dist[i][j] = min(dist[i][j], dist[i][k] + dist[k][j])
    
    return dist

def topological_sort(graph: List[List[int]]) -> List[int]:
    """Kahn's algorithm for topological sorting"""
    n = len(graph)
    in_degree = [0] * n
    
    for u in range(n):
        for v in graph[u]:
            in_degree[v] += 1
    
    queue = deque([u for u in range(n) if in_degree[u] == 0])
    result = []
    
    while queue:
        u = queue.popleft()
        result.append(u)
        
        for v in graph[u]:
            in_degree[v] -= 1
            if in_degree[v] == 0:
                queue.append(v)
    
    return result if len(result) == n else []

# Union-Find (Disjoint Set Union)
class DSU:
    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank = [0] * n
        self.size = [1] * n
    
    def find(self, x: int) -> int:
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]
    
    def union(self, x: int, y: int) -> bool:
        px, py = self.find(x), self.find(y)
        if px == py:
            return False
        if self.rank[px] < self.rank[py]:
            px, py = py, px
        self.parent[py] = px
        self.size[px] += self.size[py]
        if self.rank[px] == self.rank[py]:
            self.rank[px] += 1
        return True
    
    def connected(self, x: int, y: int) -> bool:
        return self.find(x) == self.find(y)
    
    def get_size(self, x: int) -> int:
        return self.size[self.find(x)]

# Segment Tree
class SegmentTree:
    def __init__(self, arr: List[int]):
        self.n = len(arr)
        self.tree = [0] * (4 * self.n)
        self.build(arr, 0, 0, self.n - 1)
    
    def build(self, arr: List[int], node: int, start: int, end: int):
        if start == end:
            self.tree[node] = arr[start]
        else:
            mid = (start + end) // 2
            self.build(arr, 2 * node + 1, start, mid)
            self.build(arr, 2 * node + 2, mid + 1, end)
            self.tree[node] = self.tree[2 * node + 1] + self.tree[2 * node + 2]
    
    def update(self, idx: int, val: int, node: int = 0, start: int = 0, end: int = None):
        if end is None:
            end = self.n - 1
        if start == end:
            self.tree[node] = val
        else:
            mid = (start + end) // 2
            if idx <= mid:
                self.update(idx, val, 2 * node + 1, start, mid)
            else:
                self.update(idx, val, 2 * node + 2, mid + 1, end)
            self.tree[node] = self.tree[2 * node + 1] + self.tree[2 * node + 2]
    
    def query(self, l: int, r: int, node: int = 0, start: int = 0, end: int = None) -> int:
        if end is None:
            end = self.n - 1
        if r < start or l > end:
            return 0
        if l <= start and end <= r:
            return self.tree[node]
        mid = (start + end) // 2
        left_sum = self.query(l, r, 2 * node + 1, start, mid)
        right_sum = self.query(l, r, 2 * node + 2, mid + 1, end)
        return left_sum + right_sum

# Fenwick Tree (Binary Indexed Tree)
class FenwickTree:
    def __init__(self, n: int):
        self.n = n
        self.tree = [0] * (n + 1)
    
    def update(self, i: int, delta: int):
        i += 1
        while i <= self.n:
            self.tree[i] += delta
            i += i & (-i)
    
    def query(self, i: int) -> int:
        i += 1
        s = 0
        while i > 0:
            s += self.tree[i]
            i -= i & (-i)
        return s
    
    def range_query(self, l: int, r: int) -> int:
        return self.query(r) - (self.query(l - 1) if l > 0 else 0)

# Trie (Prefix Tree)
class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_end = False
        self.count = 0

class Trie:
    def __init__(self):
        self.root = TrieNode()
    
    def insert(self, word: str):
        node = self.root
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
            node.count += 1
        node.is_end = True
    
    def search(self, word: str) -> bool:
        node = self.root
        for char in word:
            if char not in node.children:
                return False
            node = node.children[char]
        return node.is_end
    
    def starts_with(self, prefix: str) -> bool:
        node = self.root
        for char in prefix:
            if char not in node.children:
                return False
            node = node.children[char]
        return True
    
    def count_prefix(self, prefix: str) -> int:
        node = self.root
        for char in prefix:
            if char not in node.children:
                return 0
            node = node.children[char]
        return node.count

# String utilities
def z_algorithm(s: str) -> List[int]:
    """Z-array: z[i] = length of longest substring starting from s[i] which is also prefix of s"""
    n = len(s)
    z = [0] * n
    z[0] = n
    l, r = 0, 0
    
    for i in range(1, n):
        if i <= r:
            z[i] = min(r - i + 1, z[i - l])
        while i + z[i] < n and s[z[i]] == s[i + z[i]]:
            z[i] += 1
        if i + z[i] - 1 > r:
            l, r = i, i + z[i] - 1
    
    return z

def kmp_pattern_search(text: str, pattern: str) -> List[int]:
    """KMP pattern matching algorithm"""
    def compute_lps(pattern: str) -> List[int]:
        m = len(pattern)
        lps = [0] * m
        length = 0
        i = 1
        
        while i < m:
            if pattern[i] == pattern[length]:
                length += 1
                lps[i] = length
                i += 1
            else:
                if length != 0:
                    length = lps[length - 1]
                else:
                    lps[i] = 0
                    i += 1
        
        return lps
    
    n, m = len(text), len(pattern)
    if m == 0:
        return []
    
    lps = compute_lps(pattern)
    positions = []
    i = j = 0
    
    while i < n:
        if text[i] == pattern[j]:
            i += 1
            j += 1
        
        if j == m:
            positions.append(i - j)
            j = lps[j - 1]
        elif i < n and text[i] != pattern[j]:
            if j != 0:
                j = lps[j - 1]
            else:
                i += 1
    
    return positions

def rolling_hash(s: str, base: int = 31, mod: int = 10**9 + 7) -> List[int]:
    """Compute rolling hash for all prefixes"""
    n = len(s)
    hash_vals = [0] * (n + 1)
    pow_base = [1] * (n + 1)
    
    for i in range(n):
        hash_vals[i + 1] = (hash_vals[i] * base + ord(s[i])) % mod
        pow_base[i + 1] = (pow_base[i] * base) % mod
    
    return hash_vals, pow_base

def get_substring_hash(hash_vals: List[int], pow_base: List[int], l: int, r: int, mod: int = 10**9 + 7) -> int:
    """Get hash of substring s[l:r+1]"""
    return (hash_vals[r + 1] - hash_vals[l] * pow_base[r - l + 1]) % mod

# Main solver function
def solve():
    # Your solution code here
    
    arr = read_ints()
    
    # Example: Find sum of array
    result = sum(arr)
    print(result)

# Main execution
def main():
    # For single test case
    t=read_int()
    for _ in range(t):
     solve()
    
    # For multiple test cases (uncomment if needed)
    # t = read_int()
    # for _ in range(t):
    #     solve()

if __name__ == "__main__":
    main()
    
    # Close files if using file I/O
    if LOCAL_TESTING:
        sys.stdin.close()
        # sys.stdout.close()  # Uncomment if using output file