/**
 * Seed for the execution engine.
 *
 * There is nothing content-like to seed any more — no problems, no test cases.
 * All this does is guarantee two accounts exist so the playground is usable
 * immediately after `migrate deploy`.
 */
import { PrismaClient } from '@prisma/client';
import { Role } from '../src/lib/enums';
import bcrypt from 'bcryptjs';

const prisma = new PrismaClient();

const SALT_ROUNDS = Number(process.env.BCRYPT_SALT_ROUNDS ?? 10);

async function seedUsers() {
  const [adminHash, userHash] = await Promise.all([
    bcrypt.hash('Admin123!', SALT_ROUNDS),
    bcrypt.hash('Password123!', SALT_ROUNDS),
  ]);

  const admin = await prisma.user.upsert({
    where: { email: 'admin@example.com' },
    update: { role: Role.ADMIN, passwordHash: adminHash },
    create: {
      email: 'admin@example.com',
      username: 'admin',
      passwordHash: adminHash,
      role: Role.ADMIN,
    },
  });

  const researcher = await prisma.user.upsert({
    where: { email: 'coder@example.com' },
    update: { passwordHash: userHash },
    create: {
      email: 'coder@example.com',
      username: 'coder',
      passwordHash: userHash,
      role: Role.USER,
    },
  });

  return { admin, researcher };
}

async function main() {
  const { admin, researcher } = await seedUsers();
  console.log(`  user: ${admin.email} [ADMIN]`);
  console.log(`  user: ${researcher.email} [USER]`);
  console.log('Seed complete.');
}

main()
  .catch((error) => {
    console.error(error);
    process.exit(1);
  })
  .finally(() => prisma.$disconnect());
