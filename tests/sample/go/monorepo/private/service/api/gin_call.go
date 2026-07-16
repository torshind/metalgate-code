package api

import (
	"github.com/gin-gonic/gin"
)

// NewGinApp creates a gin engine using the 3rd-party gin package.
// This calls gin.New(), a qualified call to a 3rd-party package function.
// Mirrors gin's external.go calling gin.New().
func NewGinApp() *gin.Engine {
	return gin.New()
}

// NewGinDefault uses gin.Default, another qualified 3rd-party call.
func NewGinDefault() *gin.Engine {
	return gin.Default()
}
