package api

import (
	"go.example.dev/private/service/internal/shared"
)

// Controller handles API requests.
type Controller struct{}

// NewController creates a new Controller.
func NewController() *Controller {
	return &Controller{}
}

// Publish uses ToContext to build response context.
func (c *Controller) Publish(key string, val int) map[string]string {
	return shared.ToContext(key, val)
}

// Lookup uses FromContext to read request context.
func (c *Controller) Lookup(ctx map[string]string, key string) (string, error) {
	return shared.FromContext(ctx, key)
}
