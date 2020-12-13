def quick_sort(left,right,nums):
    if left<right:
        pivot = partition(left,right,nums)
        quick_sort(left,pivot-1,nums)
        quick_sort(pivot+1,right,nums)
    return nums
def partition(left,right,nums):
    pivot = nums[left]
    while left<right:
        while left<right and nums[right]>=pivot:
            right-=1
        nums[left]=nums[right]
        while left<right and nums[left]<=pivot:
            left+=1
        nums[right]=nums[left]
    nums[left]=pivot
    return left
if __name__=="__main__":
    nums = [8,4,3,5,6,0,2]
    nums=quick_sort(0,len(nums)-1,nums)
    print(nums)