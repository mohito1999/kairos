'use client'

import Link from "next/link"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { signUp } from "@/app/signup/actions"
import { useSearchParams } from 'next/navigation'

export default function SignUpForm() {
  const searchParams = useSearchParams()
  const message = searchParams.get('message')

  return (
    <div className="mx-auto w-full max-w-sm">
      <div className="text-center mb-8">
        <h1 className="text-3xl font-bold">Create an Account</h1>
        <p className="text-muted-foreground" style={{paddingTop: '10px'}}>
          Enter your information to get started with Kairos
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

          <Button formAction={signUp} type="submit" className="w-full">
            Create Account
          </Button>
        </div>
      </form>

      <div className="mt-6 text-center text-sm">
        <p>
          Already have an account?{" "}
          <Link href="/login" className="underline font-medium">
            Login
          </Link>
        </p>
      </div>
    </div>
  )
}