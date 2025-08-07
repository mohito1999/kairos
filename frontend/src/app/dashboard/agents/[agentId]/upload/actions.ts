'use server'

import { createClient } from '@/lib/supabase/server'
import { revalidatePath } from 'next/cache'
import { redirect } from 'next/navigation'

export async function uploadHistoricalData(agentId: string, formData: FormData) {
  const supabase = createClient()
  const { data: { session } } = await supabase.auth.getSession()

  if (!session) {
    return { error: 'Not authenticated' }
  }

  const file = formData.get('file') as File
  const dataMapping = formData.get('dataMapping') as string

  if (!file || !dataMapping) {
    return { error: 'File and data mapping are required.' }
  }

  // We need to create a new FormData object to send to the backend
  const backendFormData = new FormData()
  backendFormData.append('file', file)
  backendFormData.append('agent_id', agentId)
  backendFormData.append('data_mapping', dataMapping)

  try {
    const response = await fetch(`${process.env.NEXT_PUBLIC_BACKEND_URL}/api/v1/historical-data/upload`, {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${session.access_token}`,
      },
      body: backendFormData,
    })

    if (!response.ok) {
      const errorBody = await response.json()
      return { error: errorBody.detail || 'Failed to upload file.' }
    }

  } catch (error) {
    console.error("Upload error:", error)
    return { error: 'An unexpected error occurred.' }
  }

  // On success, redirect back to the agent's main page
  revalidatePath(`/dashboard/agents/${agentId}`)
  redirect(`/dashboard/agents/${agentId}`)
}