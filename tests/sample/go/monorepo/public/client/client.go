package client

import (
	"go.example.dev/private/service/api"
)

// Client wraps the API controller.
type Client struct {
	ctrl *api.Controller
}

// NewClient creates a client.
func NewClient() *Client {
	return &Client{ctrl: api.NewController()}
}

// Do calls Publish on the controller.
func (c *Client) Do(key string, val int) map[string]string {
	return c.ctrl.Publish(key, val)
}
