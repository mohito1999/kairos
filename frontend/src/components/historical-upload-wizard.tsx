'use client'

import { useState } from 'react'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { uploadHistoricalData } from '@/app/dashboard/agents/[agentId]/upload/actions'
import { useRouter } from 'next/navigation'
import { parseCsvHeaders } from '@/lib/csvHelper'

interface HistoricalUploadWizardProps {
  agentId: string
}

export default function HistoricalUploadWizard({ agentId }: HistoricalUploadWizardProps) {
  const [file, setFile] = useState<File | null>(null)
  const [headers, setHeaders] = useState<string[]>([])
  const [mapping, setMapping] = useState<{ [key: string]: string }>({})
  const [error, setError] = useState<string | null>(null)
  const [isUploading, setIsUploading] = useState(false)
  const router = useRouter()

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const selectedFile = e.target.files?.[0]
    if (selectedFile) {
      setFile(selectedFile)
      // Parse CSV headers
      const reader = new FileReader()
      reader.onload = (event) => {
        const text = event.target?.result as string;
        // CRITICAL FIX: Get the first line and remove any trailing carriage return.
        const firstLine = text.split('\n')[0].replace(/\r$/, ''); 
        
        // This is the call to our robust parser in csvHelper.ts
        setHeaders(parseCsvHeaders(firstLine));
      };
      reader.readAsText(selectedFile)
    }
  }

  const handleMappingChange = (kairosField: string, fileColumn: string) => {
    setMapping(prev => ({ ...prev, [kairosField]: fileColumn }))
  }

  const handleSubmit = async (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault()
    if (!file) {
      setError("Please select a file.")
      return
    }
    setIsUploading(true)
    setError(null)

    const formData = new FormData()
    formData.append('file', file)
    formData.append('dataMapping', JSON.stringify(mapping))

    await uploadHistoricalData(agentId, formData)
    // The server action handles redirection, but we could add a success message here
    // For simplicity, we rely on the redirect.
  }

  const kairosFields = [
    { key: "conversation_transcript", label: "Conversation Transcript" },
    { key: "outcome", label: "Outcome (e.g., 'success', 'failure')" },
    { key: "context_customer_type", label: "Context: Customer Type" },
    { key: "context_region", label: "Context: Region" },
  ]

  return (
    <Card>
      <CardHeader>
        <CardTitle>Bootstrap Agent Intelligence</CardTitle>
        <CardDescription>Upload historical call or chat logs (CSV format) to give your agent a head start.</CardDescription>
      </CardHeader>
      <CardContent>
        <form onSubmit={handleSubmit} className="space-y-8">
          <div>
            <Label htmlFor="file-upload">Step 1: Upload a CSV File</Label>
            <Input id="file-upload" type="file" accept=".csv" onChange={handleFileChange} required />
          </div>

          {headers.length > 0 && (
            <div>
              <Label>Step 2: Map Your Columns to Kairos Fields</Label>
              <div className="space-y-4 mt-2 p-4 border rounded-md">
                {kairosFields.map(field => (
                  <div key={field.key} className="grid grid-cols-2 items-center gap-4">
                    <Label>{field.label}</Label>
                    <Select onValueChange={(value) => handleMappingChange(field.key, value)}>
                      <SelectTrigger>
                        <SelectValue placeholder="Select a column from your file" />
                      </SelectTrigger>
                      <SelectContent>
                        {headers.map(header => (
                          <SelectItem key={header} value={header}>{header}</SelectItem>
                        ))}
                      </SelectContent>
                    </Select>
                  </div>
                ))}
              </div>
            </div>
          )}

          {error && <p className="text-red-500 text-sm text-center">{error}</p>}

          <Button type="submit" disabled={isUploading || headers.length === 0}>
            {isUploading ? "Uploading..." : "Upload and Process Data"}
          </Button>
        </form>
      </CardContent>
    </Card>
  )
}