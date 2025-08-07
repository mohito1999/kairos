'use client'

import Link from "next/link"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { signIn } from "@/app/login/actions"
import { useSearchParams } from 'next/navigation'

export default function LoginForm() {
  const searchParams = useSearchParams()
  const message = searchParams.get('message')

  return (
    // By removing the Card component, we get a cleaner, more direct layout.
    <div className="mx-auto w-full max-w-sm">
      <div className="text-center mb-8">
        <h1 className="text-3xl font-bold">Login</h1>
        <p className="text-muted-foreground" style={{paddingTop: '10px'}}>
          Enter your email below to login to your account
        </p>
      </div>
      
      <form>
        <div className="grid gap-6">
          <div className="grid gap-2 text-left">
            <Label htmlFor="email">Email</Label>
            <Input
              id="email"
              name="email"
              type="email"
              placeholder="you@example.com"
              required
            />
          </div>
          <div className="grid gap-2 text-left">
            <Label htmlFor="password">Password</Label>
            <Input id="password" name="password" type="password" required />
          </div>

          {message && (
            <p className="p-4 bg-muted text-muted-foreground text-center rounded-lg text-sm">
              {message}
            </p>
          )}

          <Button formAction={signIn} type="submit" className="w-full">
            Login
          </Button>
        </div>
      </form>

      {/* CRITICAL FIX: Moved "Forgot Password" and restructured the bottom links */}
      <div className="mt-6 text-center text-sm">
        <p className="mb-2">
          Don&apos;t have an account?{" "}
          <Link href="/signup" className="underline font-medium">
            Sign up
          </Link>
        </p>
        <Link
          href="#" // Future forgot password page
          className="underline text-muted-foreground hover:text-primary"
        >
          Forgot your password?
        </Link>
      </div>
    </div>
  )
}