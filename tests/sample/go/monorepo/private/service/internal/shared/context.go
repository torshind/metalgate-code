package shared

import "fmt"

// ToContext converts data to a context map.
func ToContext(key string, value int) map[string]string {
	return map[string]string{key: fmt.Sprintf("%d", value)}
}

// UnusedFunc is never called by anyone.
func UnusedFunc() {}

// FromContext extracts a value from a context map.
func FromContext(ctx map[string]string, key string) (string, error) {
	v, ok := ctx[key]
	if !ok {
		return "", fmt.Errorf("key %q not found", key)
	}
	return v, nil
}
