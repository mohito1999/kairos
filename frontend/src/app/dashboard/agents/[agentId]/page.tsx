import { createClient } from "@/lib/supabase/server"
import { cookies } from "next/headers"
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card"
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table"
import Link from "next/link"
import { Button } from "@/components/ui/button"
import { PerformanceChart } from "@/components/performance-chart"
import { Bot, Activity, ArrowUpRight, BarChart } from "lucide-react"

// Define types for our data for type safety
interface Agent {
  id: string;
  name: string;
  objective: string;
  created_at: string;
}

interface LearnedPattern {
  id: string;
  source: string;
  trigger_context_summary: string;
  suggested_strategy: string;
  status: string;
  impressions: number;
  success_rate: number;
}

// Server function to fetch a single agent's details
async function getAgentDetails(agentId: string): Promise<Agent | null> {
  const cookieStore = cookies()
  const supabase = createClient()
  const { data: { session } } = await supabase.auth.getSession()

  if (!session) return null

  try {
    const response = await fetch(`${process.env.NEXT_PUBLIC_BACKEND_URL}/api/v1/agents/${agentId}`, {
      headers: { 'Authorization': `Bearer ${session.access_token}` },
      cache: 'no-store',
    })
    if (!response.ok) return null
    return await response.json()
  } catch (error) {
    console.error("Error fetching agent details:", error)
    return null
  }
}

// Server function to fetch an agent's learned patterns
async function getAgentPatterns(agentId: string): Promise<LearnedPattern[]> {
    const cookieStore = cookies()
    const supabase = createClient()
    const { data: { session } } = await supabase.auth.getSession()
  
    if (!session) return []
  
    try {
      // This now calls the real analytics endpoint we built in the backend
      const response = await fetch(`${process.env.NEXT_PUBLIC_BACKEND_URL}/api/v1/analytics/patterns/${agentId}`, {
        headers: { 'Authorization': `Bearer ${session.access_token}` },
        cache: 'no-store',
      })
  
      if (!response.ok) {
        console.error("Failed to fetch patterns:", await response.text())
        return []
      }
      return await response.json()
    } catch (error) {
      console.error("Error fetching patterns:", error)
      return []
    }
  }

// Define types for our new data
interface PerformanceDataPoint {
    date: string;
    success_rate: number;
  }
  
  interface PerformanceAnalytics {
    timeseries: PerformanceDataPoint[];
  }
  
  // Server function to fetch performance analytics
  async function getAgentPerformance(agentId: string): Promise<PerformanceAnalytics> {
    const cookieStore = cookies()
    const supabase = createClient()
    const { data: { session } } = await supabase.auth.getSession()
  
    if (!session) return { timeseries: [] }
  
    try {
      const response = await fetch(`${process.env.NEXT_PUBLIC_BACKEND_URL}/api/v1/analytics/performance-over-time/${agentId}`, {
        headers: { 'Authorization': `Bearer ${session.access_token}` },
        cache: 'no-store',
      })
  
      if (!response.ok) {
        console.error("Failed to fetch performance data:", await response.text())
        return { timeseries: [] }
      }
      return await response.json()
    } catch (error) {
      console.error("Error fetching performance data:", error)
      return { timeseries: [] }
    }
  }

export default async function AgentDetailPage({ params }: { params: { agentId: string } }) {
  const { agentId } = params;
  // Fetch all data in parallel for better performance
  const [agent, patterns, performance] = await Promise.all([
    getAgentDetails(agentId),
    getAgentPatterns(agentId),
    getAgentPerformance(agentId),
  ])

  if (!agent) {
    return (
      <div className="flex flex-1 items-center justify-center rounded-lg border border-dashed shadow-sm">
        <div className="flex flex-col items-center gap-1 text-center">
          <h3 className="text-2xl font-bold tracking-tight">Agent Not Found</h3>
          <p className="text-sm text-muted-foreground">
            The agent you are looking for does not exist or you do not have permission to view it.
          </p>
          <Button asChild className="mt-4">
            <Link href="/dashboard/agents">Back to Agents</Link>
          </Button>
        </div>
      </div>
    )
  }
  
  // Calculate some summary stats for the cards
  const totalPatterns = patterns.length
  const activePatterns = patterns.filter(p => p.status === 'ACTIVE').length
  const latestSuccessRate = performance.timeseries.length > 0
    ? performance.timeseries[performance.timeseries.length - 1].success_rate
    : 0;

  return (
    <div className="flex flex-col gap-8">
      <div>
        <h1 className="text-3xl font-bold tracking-tight">{agent.name}</h1>
        <p className="text-muted-foreground mt-2">{agent.objective}</p>
      </div>

      <div className="grid gap-4 md:grid-cols-2 md:gap-8 lg:grid-cols-4">
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Current Success Rate</CardTitle>
            <BarChart className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{latestSuccessRate.toFixed(1)}%</div>
            <p className="text-xs text-muted-foreground">Based on recent interactions</p>
          </CardContent>
        </Card>
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Active Patterns</CardTitle>
            <Bot className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{activePatterns}</div>
            <p className="text-xs text-muted-foreground">of {totalPatterns} total learned patterns</p>
          </CardContent>
        </Card>
        {/* Add more cards here later */}
      </div>

      <div>
        <Card>
          <CardHeader>
            <CardTitle>Performance Over Time</CardTitle>
            <CardDescription>
              Success rate of the agent over the last 30 days.
            </CardDescription>
          </CardHeader>
          <CardContent>
            <PerformanceChart data={performance.timeseries} />
          </CardContent>
        </Card>
      </div>

      <div>
        <Card>
          <CardHeader>
            <CardTitle>Learned Patterns</CardTitle>
            <CardDescription>
              These are the conversational strategies Kairos has discovered or learned.
            </CardDescription>
          </CardHeader>
          <CardContent>
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>Trigger</TableHead>
                  <TableHead>Strategy</TableHead>
                  <TableHead>Source</TableHead>
                  <TableHead>Status</TableHead>
                  <TableHead>Performance</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {patterns.length === 0 ? (
                  <TableRow>
                    <TableCell colSpan={5} className="h-24 text-center">
                      No patterns learned yet. Start running interactions.
                    </TableCell>
                  </TableRow>
                ) : (
                  patterns.map((pattern) => (
                    <TableRow key={pattern.id}>
                      <TableCell>{pattern.trigger_context_summary}</TableCell>
                      <TableCell>{pattern.suggested_strategy}</TableCell>
                      <TableCell>{pattern.source}</TableCell>
                      <TableCell>{pattern.status}</TableCell>
                      <TableCell>{pattern.success_rate.toFixed(1)}% ({pattern.impressions} impressions)</TableCell>
                    </TableRow>
                  ))
                )}
              </TableBody>
            </Table>
          </CardContent>
        </Card>
      </div>
    </div>
  )
}