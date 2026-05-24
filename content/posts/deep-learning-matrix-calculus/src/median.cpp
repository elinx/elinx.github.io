#include <cassert>
#include <climits>
#include <cstddef>
#include <vector>

using std::vector;

const static unsigned NUMBERS_PER_BUCKET = 1000;

inline size_t bucketIndex(int value, int numsPerBucket) {
  return (value - INT_MIN) / numsPerBucket;
}

// assume size of stream N, size of memory M, M << N
int findMedian(const vector<int> &stream, vector<int> &memory) {
  size_t N = stream.size();
  size_t M = memory.size();
  size_t BUCKETS_NUM = UINT_MAX / NUMBERS_PER_BUCKET + 1;
  // adjust the bucket size according to the memory size if failed assertion
  assert(BUCKETS_NUM <= M);
  // return the lower bound value for even number of N for simplicity
  size_t medianIndex = N / 2;

  // iterate the stream, for each value calcualte which bucket it
  // should be putted into, only remember how many numbers in each bucket
  for (int value : stream) {
    memory[bucketIndex(value, NUMBERS_PER_BUCKET)]++;
  }

  // get bucket index where the median resided
  size_t size = 0;
  size_t targetBucketIndex = 0;
  size_t numsBeforeTargetBucket = 0;
  for (size_t i = 0; i < BUCKETS_NUM; ++i) {
    size += memory[i];
    if (size > medianIndex) {
      targetBucketIndex = i;
      numsBeforeTargetBucket = size - memory[i];
      break;
    }
  }

  // collect values from the target bucket, sort to get the target indexed value
  memory.clear();
  for (int value : stream) {
    if (bucketIndex(value, NUMBERS_PER_BUCKET) == targetBucketIndex) {
      memory.push_back(value);
    }
  }
  sort(memory.begin(), memory.end());
  return *(memory.begin() + medianIndex - numsBeforeTargetBucket);
}