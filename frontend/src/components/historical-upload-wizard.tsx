'use client'

import { useState, useId } from 'react'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { RadioGroup, RadioGroupItem } from "@/components/ui/radio-group"
import { Textarea } from '@/components/ui/textarea'
import { uploadHistoricalData } from '@/app/dashboard/agents/[agentId]/upload/actions'
import { parseCsvHeaders } from '@/lib/csvHelper'
import { Trash2 } from 'lucide-react'

interface HistoricalUploadWizardProps {
  agentId: string
}

type OutcomeMethod = 'column' | 'ai_judge';

// Type for our dynamic context field mapping
interface ContextMapping {
  id: string; // for unique key in React
  kairosKey: string;
  fileColumn: string;
}

export default function HistoricalUploadWizard({ agentId }: HistoricalUploadWizardProps) {
  // --- STATE MANAGEMENT ---
  const [file, setFile] = useState<File | null>(null)
  const [headers, setHeaders] = useState<string[]>([])
  const [isUploading, setIsUploading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  // State for the new flexible features
  const [outcomeMethod, setOutcomeMethod] = useState<OutcomeMethod>('column')
  const [outcomeColumn, setOutcomeColumn] = useState<string>('')
  const [outcomeGoal, setOutcomeGoal] = useState<string>('')
  const [transcriptColumn, setTranscriptColumn] = useState<string>('')
  const [contextMappings, setContextMappings] = useState<ContextMapping[]>([
    { id: `ctx-${useId()}`, kairosKey: 'customer_type', fileColumn: '' }
  ]);

  const routerId = useId(); // To ensure unique IDs for new fields

  // --- HANDLER FUNCTIONS ---
  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const selectedFile = e.target.files?.[0]
    if (selectedFile) {
      setFile(selectedFile)
      const reader = new FileReader()
      reader.onload = (event) => {
        const text = event.target?.result as string
        const firstLine = text.split('\n')[0].replace(/\r$/, ''); 
        setHeaders(parseCsvHeaders(firstLine));
      }
      reader.readAsText(selectedFile)
    }
  }

  const addContextMapping = () => {
    setContextMappings([...contextMappings, { id: `ctx-${routerId}-${contextMappings.length}`, kairosKey: '', fileColumn: '' }])
  }

  const removeContextMapping = (id: string) => {
    setContextMappings(contextMappings.filter(m => m.id !== id))
  }

  const updateContextMapping = (id: string, part: 'kairosKey' | 'fileColumn', value: string) => {
    setContextMappings(contextMappings.map(m => m.id === id ? { ...m, [part]: value } : m))
  }
  
  const handleSubmit = async (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    if (!file) { setError("Please select a file."); return; }
    
    // Build the final dataMapping object
    const dataMapping: { [key: string]: string } = {};
    if (transcriptColumn) dataMapping['conversation_transcript'] = transcriptColumn;

    if (outcomeMethod === 'column') {
      if (!outcomeColumn) { setError("Please select an outcome column."); return; }
      dataMapping['outcome_column'] = outcomeColumn;
    } else {
      if (!outcomeGoal) { setError("Please provide a success goal description."); return; }
      dataMapping['outcome_goal_description'] = outcomeGoal;
    }

    contextMappings.forEach(m => {
      if (m.kairosKey && m.fileColumn) {
        dataMapping[`context_${m.kairosKey}`] = m.fileColumn;
      }
    });

    // console.log("Submitting with dataMapping:", dataMapping);

    setIsUploading(true);
    setError(null);

    const formData = new FormData();
    formData.append('file', file);
    formData.append('dataMapping', JSON.stringify(dataMapping));

    await uploadHistoricalData(agentId, formData);
  };


  return (
    <Card>
      <CardHeader>
        <CardTitle>Bootstrap Agent Intelligence</CardTitle>
        <CardDescription>Upload historical call or chat logs (CSV format) to give your agent a head start.</CardDescription>
      </CardHeader>
      <CardContent>
        <form onSubmit={handleSubmit} className="space-y-8">
          {/* --- STEP 1: FILE UPLOAD --- */}
          <div className='space-y-2'>
            <Label htmlFor="file-upload" className='text-lg font-semibold'>Step 1: Upload a CSV File</Label>
            <Input id="file-upload" type="file" accept=".csv" onChange={handleFileChange} required />
          </div>

          {headers.length > 0 && (
            <>
              {/* --- STEP 2: CORE FIELD MAPPING --- */}
              <div className='space-y-2'>
                <Label className='text-lg font-semibold'>Step 2: Map Core Fields</Label>
                <div className="space-y-4 mt-2 p-4 border rounded-md">
                  <div className="grid grid-cols-2 items-center gap-4">
                    <Label>Conversation Transcript <span className="text-red-500">*</span></Label>
                    <Select onValueChange={setTranscriptColumn} required>
                      <SelectTrigger><SelectValue placeholder="Select transcript column" /></SelectTrigger>
                      <SelectContent>{headers.map(h => <SelectItem key={h} value={h}>{h}</SelectItem>)}</SelectContent>
                    </Select>
                  </div>
                </div>
              </div>

              {/* --- STEP 3: OUTCOME MAPPING --- */}
              <div className='space-y-2'>
                <Label className='text-lg font-semibold'>Step 3: Define the Outcome</Label>
                <div className="p-4 border rounded-md">
                  <RadioGroup defaultValue="column" value={outcomeMethod} onValueChange={(v: OutcomeMethod) => setOutcomeMethod(v)}>
                    <div className="flex items-center space-x-2">
                      <RadioGroupItem value="column" id="r1" />
                      <Label htmlFor="r1">Use an existing column in my file</Label>
                    </div>
                    {outcomeMethod === 'column' && (
                      <div className="pl-6 pt-2 pb-4">
                        <Select value={outcomeColumn} onValueChange={setOutcomeColumn} required>
                          <SelectTrigger><SelectValue placeholder="Select outcome column" /></SelectTrigger>
                          <SelectContent>{headers.map(h => <SelectItem key={h} value={h}>{h}</SelectItem>)}</SelectContent>
                        </Select>
                      </div>
                    )}
                    <div className="flex items-center space-x-2">
                      <RadioGroupItem value="ai_judge" id="r2" />
                      <Label htmlFor="r2">Let Kairos determine outcomes with an AI Judge</Label>
                    </div>
                     {outcomeMethod === 'ai_judge' && (
                      <div className="pl-6 pt-2 pb-4">
                        <Textarea 
                          placeholder="Describe a successful outcome, e.g., 'The customer's issue was resolved and they were happy.'"
                          value={outcomeGoal}
                          onChange={(e) => setOutcomeGoal(e.target.value)}
                          required
                        />
                      </div>
                    )}
                  </RadioGroup>
                </div>
              </div>

              {/* --- STEP 4: CUSTOM CONTEXT MAPPING --- */}
              <div className='space-y-2'>
                <Label className='text-lg font-semibold'>Step 4: Map Custom Context Fields (Optional)</Label>
                <div className="space-y-4 mt-2 p-4 border rounded-md">
                  {contextMappings.map((mapping, index) => (
                    <div key={mapping.id} className="grid grid-cols-3 items-center gap-2">
                      <Input 
                        placeholder="Kairos Field Name (e.g., lead_score)"
                        value={mapping.kairosKey}
                        onChange={(e) => updateContextMapping(mapping.id, 'kairosKey', e.target.value)}
                      />
                      <Select onValueChange={(value) => updateContextMapping(mapping.id, 'fileColumn', value)}>
                        <SelectTrigger><SelectValue placeholder="Select file column" /></SelectTrigger>
                        <SelectContent>{headers.map(h => <SelectItem key={h} value={h}>{h}</SelectItem>)}</SelectContent>
                      </Select>
                      <Button type="button" variant="ghost" size="icon" onClick={() => removeContextMapping(mapping.id)}>
                        <Trash2 className="h-4 w-4" />
                      </Button>
                    </div>
                  ))}
                  <Button type="button" variant="outline" onClick={addContextMapping}>+ Add Context Field</Button>
                </div>
              </div>
            </>
          )}

          {error && <p className="text-red-500 text-sm text-center pt-4">{error}</p>}
          
          <Button type="submit" disabled={isUploading || !file} className="w-full">
            {isUploading ? "Uploading & Processing..." : "Start Intelligence Extraction"}
          </Button>
        </form>
      </CardContent>
    </Card>
  )
}