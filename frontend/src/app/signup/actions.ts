'use server'

import { createClient } from '@/lib/supabase/server'
import { redirect } from 'next/navigation'

export async function signUp(formData: FormData) {
  const email = formData.get('email') as string
  const password = formData.get('password') as string
  const supabase = createClient()

  const { error } = await supabase.auth.signUp({
    email,
    password,
    options: {
      // You can add additional user metadata here if needed
      // emailRedirectTo: `${origin}/auth/callback`, // Uncomment for email confirmation
    },
  })

  if (error) {
    return redirect(`/signup?message=Could not create user: ${error.message}`)
  }

  // A successful sign-up will usually require email confirmation.
  // For now, we'll redirect to a page telling them to check their email.
  // If you disable email confirmation in Supabase, you can redirect to '/dashboard'.
  return redirect('/login?message=Signup successful! Please check your email to confirm your account.')
}