import AuthLayout from "@/components/auth-layout"
import LoginForm from "@/components/login-form"

export default function LoginPage() {
  return (
    <AuthLayout 
      // We'll use a placeholder image for now
      imageUrl="https://res.cloudinary.com/dsvxdx0b9/image/upload/v1754565887/catalyst-login-page_o92v2q.png" 
      imageAlt="An abstract image of colorful, intersecting light trails representing data and intelligence."
    >
      <LoginForm />
    </AuthLayout>
  )
}