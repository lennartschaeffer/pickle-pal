import { AppShell, Stack, Title, Flex, Button } from "@mantine/core";
import React from "react";
import { FaTableTennis } from "react-icons/fa";
import { GiPickle } from "react-icons/gi";
import { Link, useLocation } from "react-router-dom";

// Custom NavLink component
const NavLink = ({ to, label }: { to: string; label: string }) => {
  const location = useLocation();
  const isActive = location.pathname === to;

  return (
    <Button
      component={Link}
      to={to}
      variant={isActive ? "filled" : "subtle"}
      fullWidth
    >
      {label}
    </Button>
  );
};

interface LayoutProps {
  children: React.ReactNode;
}
const Layout = ({ children }: LayoutProps) => {
  return (
    <AppShell
      padding="md"
      header={{ height: 70 }}
      navbar={{
        width: 300,
        breakpoint: "sm",
      }}
    >
      <AppShell.Header>
        <Flex justify="start" align="center" mx={90} mt={7}>
          <Flex justify="center" gap={10} align="center">
            <GiPickle size={40} color="green" />
            <Title size={40}>PicklePal</Title>
          </Flex>
        </Flex>
      </AppShell.Header>

      <AppShell.Navbar>
        <Stack gap="md" p="md">
          <NavLink to="/" label="Home" />
        </Stack>
      </AppShell.Navbar>

      <AppShell.Main>{children}</AppShell.Main>
    </AppShell>
  );
};

export default Layout;
