import { createClient } from "@/lib/supabase/server"
import { cookies } from "next/headers"
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table"
import { RegisterAgentDialog } from "@/components/register-agent-form"

// Define a type for our agent data for type safety
interface Agent {
  id: string;
  name: string;
  objective: string;
  created_at: string;
}

// This function will run on the server to fetch data
async function getAgents(): Promise<Agent[]> {
  const cookieStore = cookies()
  const supabase = createClient()

  const { data: { session } } = await supabase.auth.getSession()

  if (!session) {
    // This should ideally not happen if middleware is working, but it's good practice
    return []
  }

  try {
    const response = await fetch(`${process.env.NEXT_PUBLIC_BACKEND_URL}/api/v1/agents/`, {
      method: 'GET',
      headers: {
        'Authorization': `Bearer ${session.access_token}`,
      },
      // Use 'no-store' to ensure we always get the latest data
      cache: 'no-store', 
    })

    if (!response.ok) {
      console.error("Failed to fetch agents:", await response.text())
      return []
    }

    const agents: Agent[] = await response.json()
    return agents
  } catch (error) {
    console.error("Error fetching agents:", error)
    return []
  }
}

export default async function AgentsPage() {
  // Call the server function to get the data
  const agents = await getAgents()

  return (
    <div className="flex flex-col gap-4">
      <div className="flex items-center justify-between">
        <h1 className="text-lg font-semibold md:text-2xl">Agents</h1>
        <RegisterAgentDialog />
      </div>
      <div className="rounded-lg border shadow-sm">
        <Table>
          <TableHeader>
            <TableRow>
              <TableHead>Name</TableHead>
              <TableHead>Objective</TableHead>
              <TableHead>Created At</TableHead>
              <TableHead className="text-right">Actions</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {agents.length === 0 ? (
              <TableRow>
                <TableCell colSpan={4} className="h-24 text-center">
                  No agents registered yet.
                </TableCell>
              </TableRow>
            ) : (
              agents.map((agent) => (
                <TableRow key={agent.id}>
                  <TableCell className="font-medium">{agent.name}</TableCell>
                  <TableCell>{agent.objective}</TableCell>
                  <TableCell>{new Date(agent.created_at).toLocaleDateString()}</TableCell>
                  <TableCell className="text-right">
                    {/* Actions dropdown will go here */}
                  </TableCell>
                </TableRow>
              ))
            )}
          </TableBody>
        </Table>
      </div>
    </div>
  )
}