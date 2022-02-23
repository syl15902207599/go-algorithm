package algorithm

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
