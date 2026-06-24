output "instance_id" {
  description = "EC2 instance ID."
  value       = aws_instance.ddpm_training.id
}

output "public_ip" {
  description = "Public IPv4 address for SSH."
  value       = aws_instance.ddpm_training.public_ip
}

output "public_dns" {
  description = "Public DNS name for SSH."
  value       = aws_instance.ddpm_training.public_dns
}

output "ssh_command" {
  description = "SSH command for connecting to the instance."
  value       = "ssh -i ${var.ssh_private_key_path} ubuntu@${aws_instance.ddpm_training.public_ip}"
}
