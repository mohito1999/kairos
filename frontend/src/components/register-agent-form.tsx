'use client'

import { Button } from "@/components/ui/button"
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Textarea } from "@/components/ui/textarea"
import { createAgent } from "@/app/dashboard/agents/actions"
import { useState } from "react"

export function RegisterAgentDialog() {
  const [isOpen, setIsOpen] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const handleSubmit = async (formData: FormData) => {
    const result = await createAgent(formData);
    if (result.error) {
      setError(result.error);
    } else {
      setError(null);
      setIsOpen(false); // Close the dialog on success
    }
  }

  return (
    <Dialog open={isOpen} onOpenChange={setIsOpen}>
      <DialogTrigger asChild>
        <Button>Register New Agent</Button>
      </DialogTrigger>
      <DialogContent className="sm:max-w-[425px]">
        <DialogHeader>
          <DialogTitle>Register a New Agent</DialogTitle>
          <DialogDescription>
            Give your agent a name and define its primary objective. Click save when you're done.
          </DialogDescription>
        </DialogHeader>
        <form action={handleSubmit}>
          <div className="grid gap-4 py-4">
            <div className="grid grid-cols-4 items-center gap-4">
              <Label htmlFor="name" className="text-right">
                Name
              </Label>
              <Input id="name" name="name" placeholder="e.g., Sales Bot V1" className="col-span-3" required />
            </div>
            <div className="grid grid-cols-4 items-center gap-4">
              <Label htmlFor="objective" className="text-right">
                Objective
              </Label>
              <Textarea
                id="objective"
                name="objective"
                placeholder="e.g., To book as many demos as possible."
                className="col-span-3"
                required
              />
            </div>
            {error && <p className="text-red-500 text-sm col-span-4 text-center">{error}</p>}
          </div>
          <DialogFooter>
            <Button type="submit">Save Agent</Button>
          </DialogFooter>
        </form>
      </DialogContent>
    </Dialog>
  )
}