import AuthLayout from "@/components/auth-layout"
import SignUpForm from "@/components/signup-form"

export default function SignUpPage() {
  return (
    <AuthLayout 
      imageUrl="https://res.cloudinary.com/dsvxdx0b9/image/upload/v1754565887/catalyst-login-page_o92v2q.png" 
      imageAlt="An abstract image of swirling pastel colors, representing creativity and new beginnings."
    >
      <SignUpForm />
    </AuthLayout>
  )
}