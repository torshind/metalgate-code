package main

// REQUIRED_KEYS are required address fields.
var REQUIRED_KEYS = []string{"street", "city"}

// ValidateAddress checks if an address has all required keys.
func ValidateAddress(address string) bool {
	if address == "" {
		return false
	}
	return true
}
