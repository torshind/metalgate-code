package main

import "strings"

// MultiSelector exercises multiple selectors on a single line, including
// the same qualifier appearing twice (like gin's recovery.go:56).
func MultiSelector(s string) string {
	return strings.ToUpper(strings.ToLower(s))
}
