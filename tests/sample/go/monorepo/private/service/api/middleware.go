package api

// Middleware wraps the controller's Publish with a default key.
// This calls c.Publish, which is defined in controller.go (same package,
// different file).  This mirrors gin's recovery.go calling c.Next which
// is defined in context.go (same package gin, different file).
func Middleware(c *Controller) map[string]string {
	return c.Publish("middleware", 1)
}

// MiddlewareLookup calls c.Lookup, defined in controller.go (same package,
// different file).  Mirrors gin's c.Error / c.Abort pattern.
func MiddlewareLookup(c *Controller, ctx map[string]string, key string) (string, error) {
	return c.Lookup(ctx, key)
}
