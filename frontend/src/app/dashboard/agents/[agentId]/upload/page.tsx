import HistoricalUploadWizard from "@/components/historical-upload-wizard";
import Link from "next/link";
import { ChevronLeft } from "lucide-react";

export default async function UploadPage({ params }: { params: { agentId: string } }) {
    const { agentId } = params;
  
    return (
      <div className="flex flex-col gap-4">
        <Link href={`/dashboard/agents/${agentId}`} className="flex items-center gap-2 text-muted-foreground hover:text-foreground">
          <ChevronLeft className="h-4 w-4" />
          Back to Agent Details
        </Link>
        <HistoricalUploadWizard agentId={agentId} />
      </div>
    );
  }