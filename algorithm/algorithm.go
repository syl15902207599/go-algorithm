package algorithm

import (
	"math"
	"sort"
	"strconv"
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
func AddTwoNumbers(l1 *ListNode, l2 *ListNode) *ListNode {
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

// 字符串转换整数 (atoi)
// 输入：s = "   -42"
// 输出：-42
func MyAtoi(s string) int {
	i, res, f, n := 0, 0, 1, len(s)
	for i < n && s[i] == ' ' {
		i++
	}
	//标记正负号
	if i > n {
		return 0
	}
	if s[i] == '-' {
		f = -1
		i++
	} else if s[i] == '+' {
		f = 1
		i++
	}
	for i < n && s[i] >= '0' && s[i] <= '9' {
		res = res*10 + int(s[i]-'0')
		if f*res > math.MaxInt32 {
			return math.MaxInt32
		}
		if f*res < math.MinInt32 {
			return math.MinInt32
		}
		i++
	}
	return f * res
}

// 回文数
// 输入：x = 121
// 输出：true
func IsPalindrome(x int) bool {
	if x < 0 {
		return false
	}
	s := strconv.Itoa(x)
	a := 0
	for i := 0; i < len(s)/2; i++ {
		a = a*10 + x%10
		x /= 10
	}
	if len(s)%2 == 1 {
		return x/10 == a
	} else {
		return x == a
	}
}

// 盛最多水的容器
// 给定一个长度为 n 的整数数组 height 。有 n 条垂线，第 i 条线的两个端点是 (i, 0) 和 (i, height[i]) 。
// 找出其中的两条线，使得它们与 x 轴共同构成的容器可以容纳最多的水。
// 返回容器可以储存的最大水量。
// 输入：[1,8,6,2,5,4,8,3,7]
// 输出：49
func MaxArea(height []int) int {
	m, n := 0, len(height)-1
	max := 0
	for m != n {
		if height[m] > height[n] {
			if max < (n-m)*height[n] {
				max = (n - m) * height[n]
			}
			n--
		} else {
			if max < (n-m)*height[m] {
				max = (n - m) * height[m]
			}
			m++
		}
	}
	return max
}

// 整数转罗马数字
// 输入: num = 3
// 输出: "III"
func IntToRoman(num int) string {
	m := map[int]string{
		1:    "I",
		5:    "V",
		10:   "X",
		50:   "L",
		100:  "C",
		500:  "D",
		1000: "M",
	}
	a := []int{}
	for num > 0 {
		a = append(a, num%10)
		num /= 10
	}
	s := ""
	for k, v := range a {
		b := int(math.Pow10(k))
		one, five, ten := m[b], m[5*b], m[10*b]
		switch v {
		case 1:
			s = one + s
		case 2:
			s = one + one + s
		case 3:
			s = one + one + one + s
		case 4:
			s = one + five + s
		case 5:
			s = five + s
		case 6:
			s = five + one + s
		case 7:
			s = five + one + one + s
		case 8:
			s = five + one + one + one + s
		case 9:
			s = one + ten + s
		}
	}
	return s
}

// 罗马数字转整数
// 输入: s = "IX"
// 输出: 9
func RomanToInt(s string) int {
	m := map[byte]int{
		'I': 1,
		'V': 5,
		'X': 10,
		'L': 50,
		'C': 100,
		'D': 500,
		'M': 1000,
	}
	res := 0
	for i := 0; i < len(s)-1; i++ {
		if m[s[i]] < m[s[i+1]] {
			res -= m[s[i]]
		} else {
			res += m[s[i]]
		}
	}
	return res + m[s[len(s)-1]]
}

// 最长公共前缀
// 编写一个函数来查找字符串数组中的最长公共前缀。
// 如果不存在公共前缀，返回空字符串 ""。
// 输入：strs = ["flower","flow","flight"]
// 输出："fl"
func LongestCommonPrefix(strs []string) string {
	if len(strs) == 0 {
		return ""
	}
	for i := 0; i < len(strs[0]); i++ {
		for j := 0; j < len(strs); j++ {
			if i == len(strs[j]) || strs[0][i] != strs[j][i] {
				return strs[0][0:i]
			}
		}
	}
	return strs[0]
}

// 三数之和
// 给你一个包含 n 个整数的数组 nums，判断 nums 中是否存在三个元素 a，b，c ，使得 a + b + c = 0 ？请你找出所有和为 0 且不重复的三元组。
// 输入：nums = [-1,0,1,2,-1,-4]
// 输出：[[-1,-1,2],[-1,0,1]]
func ThreeSum(nums []int) [][]int {
	if len(nums) < 3 {
		return [][]int{}
	}
	l := len(nums)
	sort.Ints(nums)
	res := [][]int{}
	for i := 0; i < l; i++ {
		if i > 0 && nums[i] == nums[i-1] {
			continue
		}
		left, right := i+1, l-1
		for left < right {
			val := nums[i] + nums[left] + nums[right]
			if val > 0 {
				right--
			} else if val < 0 {
				left++
			} else {
				res = append(res, []int{nums[i], nums[left], nums[right]})
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

	return res
}

// 最接近的三数之和
// 给你一个长度为 n 的整数数组 nums 和 一个目标值 target。请你从 nums 中选出三个整数，使它们的和与 target 最接近。
// 返回这三个数的和。
// 假定每组输入只存在恰好一个解。
// 输入：nums = [-1,2,1,-4], target = 1
// 输出：2
func ThreeSumClosest(nums []int, target int) int {
	if len(nums) < 3 {
		return 0
	}
	l := len(nums)
	sort.Ints(nums)
	min, dif := 0, math.MaxInt32
	for i := 0; i < l; i++ {
		if i > 0 && nums[i] == nums[i-1] {
			continue
		}
		left, right := i+1, l-1
		for left < right {
			val := nums[i] + nums[left] + nums[right]
			if val > target {
				if dif > val-target {
					min = val
					dif = val - target
				}
				right--
			} else if val < target {
				if dif > target-val {
					min = val
					dif = target - val
				}
				left++
			} else {
				return val
			}
		}
	}
	return min
}

// 电话号码的字母组合
// 给定一个仅包含数字 2-9 的字符串，返回所有它能表示的字母组合。答案可以按 任意顺序 返回。
// 输入：digits = "23"
// 输出：["ad","ae","af","bd","be","bf","cd","ce","cf"]
func LetterCombinations(digits string) []string {
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

// 四数之和
// 给你一个由 n 个整数组成的数组 nums ，和一个目标值 target 。请你找出并返回满足下述全部条件且不重复的四元组 [nums[a], nums[b], nums[c], nums[d]] （若两个四元组元素一一对应，则认为两个四元组重复）：
// 0 <= a, b, c, d < n
// a、b、c 和 d 互不相同
// nums[a] + nums[b] + nums[c] + nums[d] == target
// 你可以按 任意顺序 返回答案 。
// 输入：nums = [1,0,-1,0,-2,2], target = 0
// 输出：[[-2,-1,1,2],[-2,0,0,2],[-1,0,0,1]]
func FourSum(nums []int, target int) [][]int {
	n := len(nums)
	answer := [][]int{}
	if n < 4 {
		return answer
	}
	sort.Ints(nums)
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

// 删除链表的倒数第 N 个结点
// 给你一个链表，删除链表的倒数第 n 个结点，并且返回链表的头结点。
// 输入：head = [1,2,3,4,5], n = 2
// 输出：[1,2,3,5]
func RemoveNthFromEnd(head *ListNode, n int) *ListNode {
	list := head
	arr := []*ListNode{}
	for list != nil {
		arr = append(arr, list)
		list = list.Next
	}
	l := len(arr)
	if l > n {
		if l-n+1 >= l {
			arr[len(arr)-n-1].Next = nil
		} else {
			arr[len(arr)-n-1].Next = arr[len(arr)-n+1]
		}
	} else if len(arr) == n {
		head = head.Next
	}
	return head
}

// 有效的括号
// 给定一个只包括 '('，')'，'{'，'}'，'['，']' 的字符串 s ，判断字符串是否有效。
func IsValid(s string) bool {
	arr := []rune{}
	for _, v := range s {
		switch v {
		case ')':
			if len(arr) > 0 && arr[len(arr)-1] == '(' {
				arr = arr[:len(arr)-1]
				continue
			} else {
				return false
			}
		case ']':
			if len(arr) > 0 && arr[len(arr)-1] == '[' {
				arr = arr[:len(arr)-1]
				continue
			} else {
				return false
			}
		case '}':
			if len(arr) > 0 && arr[len(arr)-1] == '{' {
				arr = arr[:len(arr)-1]
				continue
			} else {
				return false
			}
		}
		arr = append(arr, v)
	}
	return len(arr) == 0
}

// 合并两个有序链表
// 将两个升序链表合并为一个新的 升序 链表并返回。新链表是通过拼接给定的两个链表的所有节点组成的。
// 输入：l1 = [1,2,4], l2 = [1,3,4]
// 输出：[1,1,2,3,4,4]
func MergeTwoLists(l1 *ListNode, l2 *ListNode) *ListNode {
	res := &ListNode{}
	head := res
	for l1 != nil || l2 != nil {
		if l1 == nil {
			head.Next = l2
			break
		}
		if l2 == nil {
			head.Next = l1
			break
		}
		if l1.Val > l2.Val {
			head.Next = &ListNode{l2.Val, nil}
			l2 = l2.Next
		} else {
			head.Next = &ListNode{l1.Val, nil}
			l1 = l1.Next
		}
		head = head.Next
	}
	return res.Next
}

// 括号生成
// 数字 n 代表生成括号的对数，请你设计一个函数，用于能够生成所有可能的并且 有效的 括号组合。
func GenerateParenthesis(n int) []string {
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

// 合并K个升序链表
// 给你一个链表数组，每个链表都已经按升序排列。
// 请你将所有链表合并到一个升序链表中，返回合并后的链表。
// 输入：lists = [[1,4,5],[1,3,4],[2,6]]
// 输出：[1,1,2,3,4,4,5,6]
func MergeKLists(lists []*ListNode) *ListNode {
	if len(lists) == 0 {
		return nil
	}
	arr := []int{}
	for i := 0; i < len(lists); i++ {
		for lists[i] != nil {
			arr = append(arr, lists[i].Val)
			lists[i] = lists[i].Next
		}
	}
	sort.Ints(arr)
	head := &ListNode{}
	res := head
	for i := 0; i < len(arr); i++ {
		head.Next = &ListNode{arr[i], nil}
		head = head.Next
	}
	return res.Next
}

// 两两交换链表中的节点
// 给你一个链表，两两交换其中相邻的节点，并返回交换后链表的头节点。你必须在不修改节点内部的值的情况下完成本题（即，只能进行节点交换）。
// 输入：head = [1,2,3,4]
// 输出：[2,1,4,3]
func SwapPairs(head *ListNode) *ListNode {
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

// K 个一组翻转链表
// 给你一个链表，每 k 个节点一组进行翻转，请你返回翻转后的链表。
// k 是一个正整数，它的值小于或等于链表的长度。
// 如果节点总数不是 k 的整数倍，那么请将最后剩余的节点保持原有顺序。
// 输入：head = [1,2,3,4,5], k = 3
// 输出：[3,2,1,4,5]
func ReverseKGroup(head *ListNode, k int) *ListNode {
	a := []int{}
	for head != nil {
		a = append(a, head.Val)
		head = head.Next
	}
	res := &ListNode{}
	head = res
	l := len(a)
	for i := 0; i < l; i++ {
		if l/k > i/k {
			res.Next = &ListNode{a[i/k*k+k-i%k-1], nil}
		} else {
			res.Next = &ListNode{a[i], nil}
		}
		res = res.Next
	}
	return head.Next
}
func ReverseKGroup1(head *ListNode, k int) *ListNode {
	res := &ListNode{Next: head}
	pre := res
	for pre != nil {
		tail := pre.Next
		for i := 0; i < k; i++ {
			tail = pre.Next
			if tail == nil {
				return res.Next
			}
		}
		next := tail.Next
		head, tail = reverseList(head, tail)
		pre = next
		tail.Next = head
	}
	return res.Next
}
func reverseList(head, tail *ListNode) (*ListNode, *ListNode) {
	prev := tail.Next
	p := head
	for prev != tail {
		nex := p.Next
		p.Next = prev
		prev = p
		p = nex
	}
	return tail, head
}
