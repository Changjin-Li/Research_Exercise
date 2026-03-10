# 水塘抽样（Reservoir Sampling）
+ 问题定义：当内存无法加载全部数据时，如何从包含未知大小的数据流中随机选取 $k$ 个数据，并且要保证每个数据被抽取到的概率相等。
+ 算法思路：假设数据流中的数据是一个接一个读入的，在读入第 $i$ 个数据时，如果 $i \leq k$ 则留下来，如果 $i>k$ 则以 $\frac{k}{i}$ 的概率留下来，并以 $\frac{1}{k}$ 的概率随机替换掉现有的一个数据。
+ 原理分析：
  + 在读入第 $i$ 个数据时，如果 $i \leq k$ 则每个数据留下的概率都应该是 $1$，即都留下来；
  + 如果 $i>k$，则每个数据留下的概率都应该是 $\frac{k}{i}$，
    + 当读入第 $k+1$ 个数据时，第 $k+1$ 个数据留下的概率为 $\frac{k}{i}=\frac{k}{k+1}$，留下来的 $k$ 个数据留下来的概率为 $1-\frac{k}{k+1}\cdot \frac{1}{k}=\frac{k}{k+1}$，成立；
    + 当读入第 $i>k+1$ 个数据时，第 $i$ 个数据留下的概率为 $\frac{k}{i}$，前 $i-1$ 个数据在第 $i$ 个数据到来之前留下的概率为 $\frac{k}{i-1}$，到来之后留下的概率为 $\frac{k}{i-1}\cdot(1-\frac{k}{i}\cdot \frac{1}{k})=\frac{k}{i}$，成立。
+ 代码实现：
```cpp
vector<int> ReservoirSampling(vector<int>& nums, int k)
{
    vector<int> results;
    
    for (int i = 0; i < k; i++) {
        results.push_back(nums[i]);         // 前 k 个数据直接存入结果
    }

    for (int i = k; i < nums.size(); i++) {
        int random = rand() % i;
        if (random < k) {                  // 以 k/i 的概率留下
            results[random] = nums[i];     // 以 1/k 的概率替换掉现有的一个数据
        }
    }

    return results;
}
```