package algorithm

import (
	"math"
	"sort"
	"strings"
)

//链表结构体
type ListNode struct {
	Val  int
	Next *ListNode
}

// 2数之和
// 给定一个整数数组 nums 和一个整数目标值 target，请你在该数组中找出 和为目标值 target  的那 两个 整数，并返回它们的数组下标。
// 你可以假设每种输入只会对应一个答案。但是，数组中同一个元素在答案里不能重复出现。
// 你可以按任意顺序返回答案。
// 输入：nums = [2,7,11,15], target = 9
// 输出：[0,1]
func TwoSum(nums []int, target int) []int {
	a := make(map[int]int)
	for k, v := range nums {
		_, ok := a[target-v]
		if ok {
			return []int{a[target-v], k}
		}
		a[v] = k
	}
	return []int{}
}

// 两数相加
// 给你两个 非空 的链表，表示两个非负的整数。它们每位数字都是按照 逆序 的方式存储的，并且每个节点只能存储 一位 数字。
// 请你将两个数相加，并以相同形式返回一个表示和的链表。
// 你可以假设除了数字 0 之外，这两个数都不会以 0 开头。
// 输入：l1 = [2,4,3], l2 = [5,6,4]
// 输出：[7,0,8]
func addTwoNumbers(l1 *ListNode, l2 *ListNode) *ListNode {
	res := &ListNode{}
	head := res
	var v int
	for l1 != nil || l2 != nil {
		if l1 == nil {
			v = l2.Val + v/10
			l2 = l2.Next
		} else if l2 == nil {
			v = l1.Val + v/10
			l1 = l1.Next
		} else {
			v = l1.Val + l2.Val + v/10
			l1 = l1.Next
			l2 = l2.Next
		}
		head.Next = &ListNode{Val: (v) % 10}
		head = head.Next
	}
	if v > 9 {
		head.Next = &ListNode{Val: (v) / 10}
	}
	return res.Next
}

// 无重复字符的最长子串
// 给定一个字符串 s ，请你找出其中不含有重复字符的 最长子串 的长度。
// 输入: s = "abcabcbb"
// 输出: 3
// 输入: s = "pwwkew"
// 输出: 3
func LengthOfLongestSubstring(s string) int {
	l := len(s)
	max, slow := 0, -1
	a := map[byte]int{}
	// for i := 0; i < l; i++ {
	// 	if k, ok := a[s[i]]; ok {
	// 		if max < len(a) {
	// 			max = len(a)
	// 		}
	// 		i = k + 1
	// 		a = map[byte]int{s[i]: i}
	// 		continue
	// 	}
	// 	a[s[i]] = i
	// }
	// if max < len(a) {
	// 	max = len(a)
	// }
	for i := 0; i < l; i++ {
		in, ok := a[s[i]]
		if ok {
			if max < len(a) {
				max = len(a)
			}
			for slow < in {
				delete(a, s[slow+1])
				slow++
			}
		}
		a[s[i]] = i
	}
	if max < len(a) {
		max = len(a)
	}
	return max
}

// 寻找两个正序数组的中位数
// 给定两个大小分别为 m 和 n 的正序（从小到大）数组 nums1 和 nums2。请你找出并返回这两个正序数组的 中位数 。
// 算法的时间复杂度应该为 O(log (m+n)) 。
// 输入：nums1 = [1,2], nums2 = [3,4]
// 输出：2.50000
func FindMedianSortedArrays(nums1 []int, nums2 []int) float64 {
	a := append(nums1, nums2...)
	sort.Ints(a)
	l := len(a)
	if l%2 == 1 {
		return float64(a[l/2])
	} else {
		return float64(a[l/2-1]+a[l/2]) / 2.000
	}
}

// 最长回文子串
// 给你一个字符串 s，找到 s 中最长的回文子串。
// 输入：s = "babad"
// 输出："bab"
// 解释："aba" 同样是符合题意的答案。
func LongestPalindrome(s string) string {
	l := len(s)
	long := ""
	slow := 0
	m := map[byte]int{}
	for i := 0; i < l; i++ {
		ind, ok := m[s[i]]
		if ok {
			for slow-1 < ind {
				delete(m, s[slow])
				slow++
			}
			if len(long) < i-slow+1 {
				long = s[slow-1 : i+1]
			}

		}
		m[s[i]] = i
	}
	if s[slow] == s[l-1] && len(long) < l-slow+1 {
		long = s[slow-1:]
	}
	return long
}

// Z 字形变换
// 输入：s = "PAYPALISHIRING", numRows = 4
// 输出："PINALSIGYAHRPI"
// 解释：
// P     I    N
// A   L S  I G
// Y A   H R
// P     I
func Convert(s string, numRows int) string {
	if numRows == 1 {
		return s
	}
	arr := make([]string, numRows)
	n := numRows*2 - 2
	for k, v := range s {
		x := k % n
		arr[min(x, n-x)] += string(v)
	}
	return strings.Join(arr, "")
}
func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

// 整数反转
// 给你一个 32 位的有符号整数 x ，返回将 x 中的数字部分反转后的结果。
// 如果反转后整数超过 32 位的有符号整数的范围 [−2^31,  2^31 − 1] ，就返回 0。
// 输入：x = -123
// 输出：-321
func Reverse(x int) int {
	res := 0
	for x != 0 {
		res = res*10 + x%10
		x = x / 10
	}
	if res > math.MaxInt32 || res < math.MinInt32 {
		return 0
	}
	return res
}
