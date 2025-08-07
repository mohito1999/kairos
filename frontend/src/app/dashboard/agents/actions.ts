'use server'

import { createClient } from '@/lib/supabase/server'
import { revalidatePath } from 'next/cache'
import { redirect } from 'next/navigation'

export async function createAgent(formData: FormData) {
  const supabase = createClient()

  const { data: { session } } = await supabase.auth.getSession()

  if (!session) {
    return {
      error: 'Not authenticated',
    }
  }

  const agentData = {
    name: formData.get('name') as string,
    objective: formData.get('objective') as string,
  }

  // This is the API call to our own FastAPI backend
  const response = await fetch(`${process.env.NEXT_PUBLIC_BACKEND_URL}/api/v1/agents/`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'Authorization': `Bearer ${session.access_token}`,
    },
    body: JSON.stringify(agentData),
  })

  if (!response.ok) {
    const errorBody = await response.json()
    console.error("Failed to create agent:", errorBody)
    return {
      error: errorBody.detail || 'Failed to create agent. Please try again.',
    }
  }
  
  // On success, we revalidate the path to refresh the agent list
  revalidatePath('/dashboard/agents')
  return {
    error: null
  }
}