package main

import (
	"fmt"
	"math"
	"sort"

	"go-algorithm.cn/algorithm"
)

func main() {
	a := algorithm.LengthOfLongestSubstring("dvdf")
	fmt.Println(a)
}

func test(g, e [4]int) string {
	gun, emm := 0, 0
	for _, v := range g {
		gun += v
	}
	for _, v := range e {
		emm += v
	}
	if gun > emm {
		return "Gunnar"
	} else if gun < emm {
		return "Emma"
	} else {
		return "Tie "
	}
}

func searchInsert(nums []int, target int) int {
	n := len(nums)
	left, right := 0, n-1
	ans := n
	for left <= right {
		mid := (right-left)/2 + left
		if target <= nums[mid] {
			ans = mid
			right = mid - 1
		} else {
			left = mid + 1
		}
	}
	return ans
}
func searchRange(nums []int, target int) []int {
	l := len(nums)
	if l == 0 {
		return []int{-1, -1}
	}
	i, j := 0, l-1
	left, right := false, false
	res := []int{}
	for i <= j {
		fmt.Printf("i:%v  j:%v\n", i, j)
		if !left {
			if nums[i] == target {
				res = append(res, i)
				left = true
			} else {
				i++
			}
		}
		if !right {
			if nums[j] == target {
				res = append(res, j)
				right = true
			} else {
				j--
			}
		}
		if left && right {
			break
		}
	}
	if len(res) == 0 {
		return []int{-1, -1}
	}
	if len(res) == 1 {
		return append(res, res[0])
	}
	sort.Ints(res)
	return res
}
func search(nums []int, target int) int {
	l := len(nums)
	if l == 0 {
		return -1
	}
	if l == 1 {
		if nums[0] == target {
			return 0
		} else {
			return -1
		}
	}
	i, j := 0, l-1
	for i <= j {
		if nums[i] == target {
			return i
		}
		if nums[j] == target {
			return j
		}
		i++
		j--
	}
	return -1
}
func longestValidParentheses(s string) int {
	res := 0
	stack := []int{-1}
	for k, v := range s {
		if v == '(' {
			stack = append(stack, k)
		} else {
			if v == ')' {
				stack = stack[:len(stack)-1]
				if len(stack) == 0 {
					stack = append(stack, k)
				} else {
					if res < k-stack[len(stack)-1] {
						res = k - stack[len(stack)-1]
					}
				}
			}
		}
	}
	return res
}

func nextPermutation(nums []int) []int {
	l := len(nums)
	if l <= 1 {
		return nums
	}
	i := l - 2
	for i >= 0 && nums[i] >= nums[i+1] {
		i--
	}
	if i >= 0 {
		j := l - 1
		for nums[i] >= nums[j] {
			j--
		}
		nums[i], nums[j] = nums[j], nums[i]
	}
	sort.Ints(nums[i+1:])
	nums = append(nums[:i+1], nums[i+1:]...)
	return nums
}

func divide(dividend int, divisor int) int {
	m := (dividend >= 0 && divisor > 0) || (dividend < 0 && divisor < 0)

	if math.MinInt32 == dividend && divisor == -1 {
		return math.MaxInt32
	}
	if dividend < 0 {
		dividend = 0 - dividend
	}
	if divisor < 0 {
		divisor = 0 - divisor
	}
	res := 0
	for i := 31; i >= 0; i-- {
		if (dividend>>i)-divisor >= 0 {
			dividend = dividend - (divisor << i)
			res += 1 << i
		}
	}
	if !m {
		return 0 - res
	}
	return res
}

func strStr(haystack string, needle string) int {
	m, n := len(haystack), len(needle)
	if n == 0 {
		return 0
	}
	j := 0
	res := -1
	for i := 0; i < m; i++ {
		if haystack[i] == needle[j] {
			if j == n-1 {
				res = i - j
				break
			}
			j++
		} else {
			i = i - j
			j = 0
		}
	}
	return res
}
func removeElement(nums []int, val int) int {
	l := len(nums)
	if l == 0 {
		return 0
	}
	slow := 0
	for i := 0; i < l; i++ {
		if nums[i] != val {
			nums[slow] = nums[i]
			slow++
		}
	}
	return slow
}
func removeDuplicates(nums []int) int {
	n := len(nums)
	if n == 0 {
		return 0
	}
	slow := 1
	for fast := 1; fast < n; fast++ {
		if nums[fast] != nums[fast-1] {
			nums[slow] = nums[fast]
			slow++
		}
	}
	return slow
}

func reverseKGroup(head *ListNode, k int) *ListNode {
	if k == 1 {
		return head
	}
	nums := []int{}
	for head != nil {
		nums = append(nums, head.Val)
		head = head.Next
	}
	tmp := &ListNode{}
	res := &ListNode{Next: tmp}
	l := len(nums)
	fmt.Println(l / k)
	for i := 0; i < l; i++ {
		fmt.Println(i / k)
		if i/k == l/k {
			tmp.Val = nums[i]
		} else {
			tmp.Val = nums[(i/k)*k+k-i%k-1]
		}
		if i == l-1 {
			break
		}
		tmp.Next = &ListNode{}
		tmp = tmp.Next
	}
	return res.Next
}

func swapPairs(head *ListNode) *ListNode {
	if head == nil || head.Next == nil {
		return head
	}
	res := head
	for head != nil && head.Next != nil {
		tmp := head.Val
		head.Val = head.Next.Val
		head.Next.Val = tmp
		head = head.Next.Next
	}
	return res
}

func mergeKLists(lists []*ListNode) *ListNode {
	nums := []int{}
	for i := 0; i < len(lists); i++ {
		if lists[i] != nil {
			for lists[i] != nil {
				nums = append(nums, lists[i].Val)
				lists[i] = lists[i].Next
			}
		}
	}
	if len(nums) == 0 {
		return nil
	}
	sort.Ints(nums)
	res := &ListNode{}
	head := res
	for j := 0; j < len(nums); j++ {
		head.Next = &ListNode{Val: nums[j]}
		head = head.Next
	}
	return res.Next
}

func generateParenthesis(n int) []string {
	res := []string{}
	var fun func(s string, lNum, rNum int)
	fun = func(s string, lNum, rNum int) {
		if len(s) == 2*n {
			res = append(res, s)
			return
		}
		if lNum > 0 {
			fun(s+"(", lNum-1, rNum)
		}
		if lNum < rNum {
			fun(s+")", lNum, rNum-1)
		}
	}
	fun("", n, n)
	return res
}

type ListNode struct {
	Val  int
	Next *ListNode
}

func removeNthFromEnd(head *ListNode, n int) *ListNode {
	tmp := head
	l := 0
	for tmp != nil {
		tmp = tmp.Next
		l++
	}
	if n == l {
		return head.Next
	}
	tmp = head
	for i := 1; i <= l; i++ {
		if i == l-n {
			tmp.Next = tmp.Next.Next
			break
		}
		tmp = tmp.Next
	}
	return head
}

func fourSum(nums []int, target int) [][]int {
	var answer [][]int
	sort.Ints(nums)
	n := len(nums)
	for i := 0; i < n-3 && nums[i]+nums[i+1]+nums[i+2]+nums[i+3] <= target; i++ {
		if i > 0 && nums[i] == nums[i-1] || nums[i]+nums[n-3]+nums[n-2]+nums[n-1] < target {
			continue
		}
		for j := i + 1; j < n-2 && nums[i]+nums[j]+nums[j+1]+nums[j+2] <= target; j++ {
			if j > i+1 && nums[j] == nums[j-1] || nums[i]+nums[j]+nums[n-2]+nums[n-1] < target {
				continue
			}
			left := j + 1
			right := n - 1
			for left < right {
				value := nums[left] + nums[right] + nums[i] + nums[j]
				if value < target {
					left++
				} else if value > target {
					right--
				} else {
					answer = append(answer, []int{nums[j], nums[i], nums[left], nums[right]})

					left++
					right--
					for left < right && nums[left] == nums[left-1] {
						left++
					}

					for left < right && nums[right] == nums[right+1] {
						right--
					}
				}
			}
		}
	}
	return answer
}

func letterCombinations(digits string) []string {
	m := map[rune][]string{
		'2': {"a", "b", "c"},
		'3': {"d", "e", "f"},
		'4': {"g", "h", "i"},
		'5': {"j", "k", "l"},
		'6': {"m", "n", "o"},
		'7': {"p", "q", "r", "s"},
		'8': {"t", "u", "v"},
		'9': {"w", "x", "y", "z"},
	}
	res := []string{}
	for _, v := range digits {
		if len(res) == 0 {
			res = m[v]
			continue
		}
		tmp := make([]string, len(res)*len(m[v]))
		for i := 0; i < len(res); i++ {
			for j := 0; j < len(m[v]); j++ {
				tmp[i*len(m[v])+j] = res[i] + m[v][j]
			}
		}
		res = tmp
	}
	return res
}

//3数之和=0
func threeSum(nums []int) [][]int {
	var answer [][]int

	sort.Ints(nums)
	for i := 0; i < len(nums)-2 && nums[i] <= 0; i++ {
		if i > 0 && nums[i] == nums[i-1] {
			continue
		}

		left := i + 1
		right := len(nums) - 1
		for left < right {
			value := nums[left] + nums[right] + nums[i]
			if value < 0 {
				left++
			} else if value > 0 {
				right--
			} else {
				answer = append(answer, []int{nums[i], nums[left], nums[right]})

				left++
				right--
				for left < right && nums[left] == nums[left-1] {
					left++
				}

				for left < right && nums[right] == nums[right+1] {
					right--
				}
			}
		}
	}
	return answer
}

// 输入：nums = [2,7,11,15], target = 9
// 输出：[0,1]
// 解释：因为 nums[0] + nums[1] == 9 ，返回 [0, 1] 。
func twoSum(nums []int, target int) []int {
	a := make(map[int]int)
	for k, v := range nums {
		if k1, ok := a[target-v]; ok {
			return []int{k, k1}
		}
		a[v] = k
	}
	return []int{}
}
