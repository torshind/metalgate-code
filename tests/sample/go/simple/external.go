package main

import "github.com/gin-gonic/gin"

// UseGin calls a function from a 3rd-party package.
func UseGin() *gin.Engine {
	return gin.New()
}
